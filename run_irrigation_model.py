"""
Single script: load sap + weather + precip, train 7-day baseline + RF model,
run live monitor (Brain 1) and forecast recommendations (Brain 2).
Integrates OpenET for ET-based irrigation amounts by zone (Low/Medium/High yield).
Run from project root: python3 run_irrigation_model.py
OpenET key: set OPENET_API_KEY in .env (copy .env.example to .env and edit).
"""
import json
import os
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

def _load_dotenv(root: Path) -> None:
    env_file = root / ".env"
    if not env_file.exists():
        return
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k:
                        os.environ.setdefault(k, v)
    except OSError:
        pass

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
_load_dotenv(ROOT)

SAP_DIR = ROOT / "sap"
WEATHER_DIR = ROOT / "weather"
PRECIP_DIR = ROOT / "precipitation"

FILE_S1 = SAP_DIR / "sap_flow_sensor1.csv"
FILE_S2 = SAP_DIR / "sap_flow_sensor2.csv"
FILE_WEATHER = WEATHER_DIR / "bondville_2025_jday182_273.csv"
FILE_PRECIP = PRECIP_DIR / "precipitation_consolidated.csv"

FORECAST_LAT = 40.05
FORECAST_LON = -88.37
USE_API_FORECAST = True

# OpenET: evapotranspiration for irrigation amount.
OPENET_API_KEY = os.environ.get("OPENET_API_KEY")
OPENET_BASE = "https://openet-api.org"
ET_LOOKBACK_DAYS = 7  # days of ET 

ZONE_IRRIGATION_FACTORS = {
    "Low Yield Zone": 0.85,
    "Medium Yield Zone": 1.0,
    "High Yield Zone": 1.15,
}

# Sensor allocation by zone: share of sensors to deploy in each zone 
# High yield = 50%, Medium = 30%, Low = 20%.
SENSOR_ALLOCATION = {
    "High Yield Zone": 0.50,
    "Medium Yield Zone": 0.30,
    "Low Yield Zone": 0.20,
}


def fetch_openet_et(
    api_key: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    interval: str = "daily",
    variable: str = "ET",
    units: str = "mm",
) -> list[tuple[str, float]]:
    """
    Fetch daily ET (mm) from OpenET raster/timeseries/point.
    Returns list of (date_str, et_mm). geometry = [longitude, latitude].
    """
    url = f"{OPENET_BASE}/raster/timeseries/point"
    body = {
        "date_range": [start_date, end_date],
        "interval": interval,
        "geometry": [lon, lat],
        "model": "Ensemble",
        "variable": variable,
        "reference_et": "gridMET",
        "units": units,
        "file_format": "JSON",
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
            out = json.load(resp)
    except Exception as e:
        raise RuntimeError(f"OpenET API failed: {e}") from e
    pairs: list[tuple[str, float]] = []
    if isinstance(out, list):
        for row in out:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                pairs.append((str(row[0])[:10], float(row[1])))
            elif isinstance(row, dict):
                d = row.get("date") or row.get("Date") or row.get("datetime")
                v = row.get("et") or row.get("ET") or row.get("value")
                if v is None and isinstance(row.get("values"), list) and row.get("values"):
                    v = row["values"][0]
                if d is not None and v is not None:
                    pairs.append((str(d)[:10], float(v)))
    data_list = []
    if isinstance(out, dict):
        data_list = out.get("data") or out.get("results") or out.get("values") or []
    if not pairs and data_list:
        for row in data_list:
            if isinstance(row, dict):
                d = row.get("date") or row.get("Date") or row.get("datetime")
                v = row.get("et") or row.get("ET") or row.get("value")
                if d is not None and v is not None:
                    pairs.append((str(d)[:10], float(v)))
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                pairs.append((str(row[0])[:10], float(row[1])))
    if not pairs and isinstance(out, dict):
        for k, v in out.items():
            if k not in ("data", "results", "values") and isinstance(v, (int, float)):
                pairs.append((str(k)[:10], float(v)))
    return pairs


def get_recent_precip_mm(precip_path: Path, reference_date: pd.Timestamp, days: int = 7) -> float:
    """Sum precipitation (mm) for the last `days` up to and including reference_date."""
    precip = pd.read_csv(precip_path)
    precip = precip[precip["record_type"] == "daily"].copy()
    month = precip["month"]
    if month.dtype == object or str(month.dtype) == "string":
        month_num = month.map({"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
                               "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12})
    else:
        month_num = month
    precip["date"] = pd.to_datetime(precip["year"].astype(str) + "-" + month_num.astype(str) + "-" + precip["day"].astype(str))
    precip["precip_mm"] = precip["total_precip_in"] * 25.4
    ref = reference_date.normalize()
    start = ref - pd.Timedelta(days=days - 1)
    mask = (precip["date"] >= start) & (precip["date"] <= ref)
    return precip.loc[mask, "precip_mm"].sum()


def compute_irrigation_by_zone(
    et_pairs: list[tuple[str, float]],
    precip_mm: float,
    zone_factors: dict[str, float],
) -> tuple[float, float, dict[str, float]]:
    """
    Deficit = sum(ET) - precip_mm over the same period. Base irrigation = max(0, deficit).
    Per-zone amounts = base_mm * zone_factors[zone]. Returns (et_total_mm, deficit_mm, zone_mm).
    """
    et_total = sum(v for _, v in et_pairs)
    deficit = max(0.0, et_total - precip_mm)
    zone_mm = {zone: round(deficit * factor, 1) for zone, factor in zone_factors.items()}
    return (round(et_total, 1), round(deficit, 1), zone_mm)


def fetch_forecast_48h_api(lat: float = FORECAST_LAT, lon: float = FORECAST_LON) -> pd.DataFrame:
    """
    Fetch next 48 hours hourly forecast from Open-Meteo (no API key).
    Returns 15-min rows for Brain 2. On SSL errors (e.g. some macOS), script falls back to mock.
    """
    hourly = "temperature_2m,relative_humidity_2m,windspeed_10m,precipitation"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": hourly,
        "timezone": "America/Chicago",
        "forecast_days": 2,
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            data = json.load(resp)
    except Exception as e:
        raise RuntimeError(f"Forecast API failed: {e}") from e
    h = data.get("hourly", {})
    if not h or "time" not in h:
        raise RuntimeError("No hourly data in forecast response")
    times = h["time"]
    n = len(times)
    #  15-min grid: each hour → 4 rows (00, 15, 30, 45)
    realdates = []
    temp = []
    rh = []
    windspd = []  
    dw_solar = []
    total_precip_mm = []
    for i in range(n):
        t = pd.Timestamp(times[i])
        for minute in [0, 15, 30, 45]:
            realdates.append(t.replace(minute=minute))
            temp.append(h["temperature_2m"][i])
            rh.append(h["relative_humidity_2m"][i])
            windspd.append(h["windspeed_10m"][i] / 3.6) 
            dw_solar.append(0.0)  
            total_precip_mm.append(h["precipitation"][i] / 4.0)
    df = pd.DataFrame({
        "realdate": realdates,
        "temp": temp,
        "rh": rh,
        "windspd": windspd,
        "dw_solar": dw_solar,
        "total_precip_mm": total_precip_mm,
    })
    return df


def load_and_merge_data(s1_path, s2_path, weather_path, precip_path):
    print("--- PREPROCESSING DATA ---")

    def process_sensor(path, sensor_name):
        df = pd.read_csv(path)
        df["realdate"] = pd.to_datetime(df["realdate"])
        if "svalue_1" in df.columns:
            df[sensor_name] = df[["svalue_1", "svalue_2", "svalue_3", "svalue_4"]].mean(axis=1)
        else:
            df[sensor_name] = df["sap_flow_mean"]
        return df[["realdate", sensor_name]].set_index("realdate").resample("15min").mean()

    s1 = process_sensor(s1_path, "s1_mean")
    s2 = process_sensor(s2_path, "s2_mean")

    sensors = pd.merge(s1, s2, left_index=True, right_index=True, how="outer")
    sensors["sap_flow_mean"] = sensors[["s1_mean", "s2_mean"]].mean(axis=1, skipna=True)

    weather = pd.read_csv(weather_path)
    weather_dates = weather[["year", "month", "day", "hour", "min"]].rename(columns={"min": "minute"})
    weather["realdate"] = pd.to_datetime(weather_dates)
    weather = weather[["realdate", "temp", "rh", "windspd", "dw_solar"]]
    weather = weather.set_index("realdate").resample("15min").mean()

    precip = pd.read_csv(precip_path)
    precip = precip[precip["record_type"] == "daily"].copy()
    month = precip["month"]
    if month.dtype == object or str(month.dtype) == "string":
        month_num = month.map({"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
                               "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12})
    else:
        month_num = month
    precip["date"] = pd.to_datetime(precip["year"].astype(str) + "-" + month_num.astype(str) + "-" + precip["day"].astype(str))
    precip["total_precip_mm"] = precip["total_precip_in"] * 25.4
    precip = precip[["date", "total_precip_mm"]].rename(columns={"date": "realdate"}).set_index("realdate")
    precip_15min = precip.resample("15min").ffill()

    df_master = pd.merge(sensors[["sap_flow_mean"]], weather, left_index=True, right_index=True, how="inner")
    df_master = df_master.join(precip_15min, how="left")
    df_master["total_precip_mm"] = df_master["total_precip_mm"].fillna(0.0)
    df_master = df_master.ffill().bfill().reset_index()

    print("Data successfully aligned to 15-minute intervals!")
    return df_master


def train_stage_1(df_master):
    print("\n--- STAGE 1: CALIBRATION & ML TRAINING ---")

    start_time = df_master["realdate"].min()
    end_time = start_time + pd.Timedelta(days=7)
    calibration_data = df_master[(df_master["realdate"] >= start_time) & (df_master["realdate"] < end_time)].copy()

    calibration_data["time_of_day"] = calibration_data["realdate"].dt.time
    baseline = calibration_data.groupby("time_of_day")["sap_flow_mean"].quantile(0.25).reset_index()
    baseline = baseline.rename(columns={"sap_flow_mean": "baseline_25th_pct"})
    baseline.to_csv(ROOT / "trained_baseline.csv", index=False)

    df_ml = df_master.copy()
    df_ml["hour"] = df_ml["realdate"].dt.hour
    df_ml["minute"] = df_ml["realdate"].dt.minute
    df_ml = df_ml.dropna()

    features = ["temp", "rh", "windspd", "dw_solar", "total_precip_mm", "hour", "minute"]
    target = "sap_flow_mean"

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df_ml[features], df_ml[target])

    joblib.dump(model, ROOT / "brain2_historical_model.pkl")
    print("Model trained on 2025 data.")
    return baseline


def run_brain_1_monitor(df_current, baseline):
    print("\n--- BRAIN 1: LIVE MONITORING ---")
    df = df_current.copy()

    df["4hr_trailing_avg"] = df["sap_flow_mean"].rolling(window=16, min_periods=1).mean()
    df["time_of_day"] = df["realdate"].dt.time
    df = pd.merge(df, baseline, on="time_of_day", how="left")

    df["is_stressed"] = df["4hr_trailing_avg"] < df["baseline_25th_pct"]

    for c in ["sap_flow_mean", "4hr_trailing_avg", "baseline_25th_pct"]:
        if c in df.columns:
            df[c] = df[c].round(10)

    latest = df.iloc[-1]
    print(f"Latest data time (dataset end): {latest['realdate']}")
    print(f"4Hr Avg: {latest['4hr_trailing_avg']} | Baseline 25th: {latest['baseline_25th_pct']}")
    print(f">> STRESSED: {latest['is_stressed']} <<")
    return df


def run_brain_2_forecast(forecast_df, current_is_stressed):
    print("\n--- BRAIN 2: FORECAST (NEXT 48 HRS) ---")

    model = joblib.load(ROOT / "brain2_historical_model.pkl")
    baseline = pd.read_csv(ROOT / "trained_baseline.csv")
    baseline["time_of_day"] = pd.to_datetime(baseline["time_of_day"], format="mixed").dt.time

    df = forecast_df.copy()
    df["realdate"] = pd.to_datetime(df["realdate"])
    df["hour"] = df["realdate"].dt.hour
    df["minute"] = df["realdate"].dt.minute

    features = ["temp", "rh", "windspd", "dw_solar", "total_precip_mm", "hour", "minute"]
    df["predicted_sap_flow"] = model.predict(df[features])

    df["future_4hr_avg"] = df["predicted_sap_flow"].rolling(window=16, min_periods=1).mean()
    df["time_of_day"] = df["realdate"].dt.time
    df = pd.merge(df, baseline, on="time_of_day", how="left")

    df["will_be_stressed"] = df["future_4hr_avg"] < df["baseline_25th_pct"]

    expected_rain_mm = df["total_precip_mm"].sum()
    future_stress_detected = df["will_be_stressed"].any()

    print(f"Expected Rain (48h): {expected_rain_mm:.1f} mm")
    print(f"Future stress: {future_stress_detected}")

    scenario = None
    print("\n--- RECOMMENDATION ---")
    if expected_rain_mm >= 10.0:
        scenario = "A"
        print("[A] Hold irrigation. Significant rainfall expected.")
    elif (current_is_stressed or future_stress_detected) and expected_rain_mm < 5.0:
        scenario = "B"
        print("[B] CRITICAL. Irrigate immediately. No significant rainfall forecasted.")
    elif (current_is_stressed or future_stress_detected) and (5.0 <= expected_rain_mm < 10.0):
        scenario = "C"
        print("[C] Pulse irrigation. Forecasted rain insufficient.")
    else:
        print("Monitor. Plant healthy, weather stable.")
    return df, scenario, expected_rain_mm, (current_is_stressed or future_stress_detected)


def create_mock_forecast(df_master):
    forecast = df_master.tail(192).copy()
    forecast["temp"] = forecast["temp"] + 5.0
    forecast["total_precip_mm"] = 0.0
    forecast.to_csv(ROOT / "mock_48hr_forecast.csv", index=False)
    return forecast


if __name__ == "__main__":
    df_historical = load_and_merge_data(FILE_S1, FILE_S2, FILE_WEATHER, FILE_PRECIP)
    baseline_df = train_stage_1(df_historical)

    current_state_df = run_brain_1_monitor(df_historical, baseline_df)
    currently_stressed = current_state_df.iloc[-1]["is_stressed"]

    if USE_API_FORECAST:
        print("\n--- FETCHING 48H FORECAST (Open-Meteo API) ---")
        try:
            forecast_df = fetch_forecast_48h_api()
            print(f"Got {len(forecast_df)} rows (15-min resolution).")
        except Exception as e:
            print(f"API forecast failed ({e}), using mock.")
            forecast_df = create_mock_forecast(df_historical)
    else:
        print("\n--- MOCK FORECAST ---")
        forecast_df = create_mock_forecast(df_historical)

    df_future, scenario, expected_rain_mm, needs_irrigation = run_brain_2_forecast(forecast_df, currently_stressed)

    reference_date = pd.Timestamp(df_historical["realdate"].max())
    if OPENET_API_KEY:
        try:
            end_str = reference_date.strftime("%Y-%m-%d")
            start_str = (reference_date - pd.Timedelta(days=ET_LOOKBACK_DAYS - 1)).strftime("%Y-%m-%d")
            et_pairs = fetch_openet_et(
                OPENET_API_KEY,
                FORECAST_LAT,
                FORECAST_LON,
                start_str,
                end_str,
                interval="daily",
                variable="ET",
                units="mm",
            )
            precip_7d = get_recent_precip_mm(FILE_PRECIP, reference_date, days=ET_LOOKBACK_DAYS)
            et_total, deficit_mm, zone_mm = compute_irrigation_by_zone(
                et_pairs, precip_7d, ZONE_IRRIGATION_FACTORS
            )
            print("\n--- HOW MUCH WATER TO IRRIGATE (OpenET + zones) ---")
            print(f"Last {ET_LOOKBACK_DAYS} days: ET = {et_total} mm, Precip = {precip_7d:.1f} mm → Deficit = {deficit_mm} mm.")
            if scenario == "A" or not needs_irrigation:
                print("No irrigation needed now. Expected rain covers or exceeds recent demand.")
            else:
                print(f"Irrigate (field-level): {deficit_mm} mm.")
                for zone_name, mm in zone_mm.items():
                    short = zone_name.replace(" Yield Zone", "")
                    print(f"  By zone — {short}: {mm} mm")
        except Exception as e:
            print(f"\nOpenET irrigation amount skipped: {e}")
    else:
        print("\nSet OPENET_API_KEY in .env (see .env.example) for irrigation amount by zone (mm).")

    print("\n--- ZONES: SENSOR ALLOCATION ---")
    for zone_name, pct in SENSOR_ALLOCATION.items():
        short = zone_name.replace(" Yield Zone", "")
        print(f"  {short}: {int(pct * 100)}% of sensors")

    print("\nDone.")
