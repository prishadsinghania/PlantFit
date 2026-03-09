#!/usr/bin/env python3
"""
Convert Illinois Climate Network (ICN) precipitation HTML files in this folder
to a single consolidated CSV. Preserves all daily data and summary rows (TOT, AVG, MAX, MIN).
"""

import re
import csv
from pathlib import Path

DATA_COLUMNS = [
    "day",
    "max_wind_speed_mph",
    "avg_wind_speed_mph",
    "avg_wind_dir_deg",
    "solar_rad_mj_m2",
    "max_air_temp_f",
    "min_air_temp_f",
    "avg_air_temp_f",
    "rel_hum_max_pct",
    "rel_hum_min_pct",
    "dew_point_f",
    "total_precip_in",
    "total_evap_in",
    "soil_under_sod_4in_max_f",
    "soil_under_sod_4in_min_f",
    "soil_under_sod_4in_avg_f",
    "soil_under_sod_8in_max_f",
    "soil_under_sod_8in_min_f",
    "soil_under_sod_8in_avg_f",
    "soil_bare_4in_max_f",
    "soil_bare_4in_min_f",
    "soil_bare_4in_avg_f",
    "soil_bare_2in_max_f",
    "soil_bare_2in_min_f",
    "soil_bare_2in_avg_f",
]


def extract_pre_content(html_path: Path) -> str:
    """Extract text from <pre> tag in HTML file."""
    text = html_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"<pre[^>]*>(.*?)</pre>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"No <pre> block found in {html_path}")
    return match.group(1).strip()


def parse_month_year(lines: list[str]) -> tuple[str, str]:
    """Find line like 'July 2025' and return (month, year)."""
    for line in lines:
        line = line.strip()
        m = re.match(r"^(\w+)\s+(\d{4})\s*$", line)
        if m and m.group(1) in (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ):
            return m.group(1), m.group(2)
    return "Unknown", ""


def parse_one_file(html_path: Path) -> list[dict]:
    """Parse one HTML file and return list of row dicts with month, year, record_type, and data columns."""
    pre = extract_pre_content(html_path)
    lines = [ln.rstrip() for ln in pre.splitlines()]

    month, year = parse_month_year(lines)

    # Find header line (starts with DAY and uses tabs)
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("DAY\t") or (line.strip().startswith("DAY") and "\t" in line):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find DAY header in {html_path}")

    start_idx = header_idx + 1
    for i in range(header_idx + 1, len(lines)):
        if re.match(r"^_+$", lines[i].strip().replace(" ", "")):
            start_idx = i + 1
            break

    rows = []
    for i in range(start_idx, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        if line.strip().startswith("KEY:") or line.strip().startswith("**"):
            break

        parts = line.split("\t")
        if not parts:
            continue

        first = parts[0].strip()
        if first.isdigit():
            record_type = "daily"
            day_val = first
        elif first in ("TOT", "AVG", "MAX", "MIN"):
            record_type = first
            day_val = ""
        else:
            continue

        row = {
            "source_file": html_path.name,
            "month": month,
            "year": year,
            "record_type": record_type,
            "day": day_val if record_type == "daily" else "",
        }

        values = [p.strip() for p in parts[1:]]
        for j, col in enumerate(DATA_COLUMNS[1:], start=0):  
            if j < len(values) and values[j]:
                try:
                    row[col] = float(values[j])
                except ValueError:
                    row[col] = values[j]
            else:
                row[col] = ""

        rows.append(row)

    return rows


def main():
    script_dir = Path(__file__).resolve().parent
    html_files = sorted(script_dir.glob("*.html")) + sorted(script_dir.glob("*.asp.html"))
    html_files = [f for f in html_files if f.name.endswith((".html", ".asp.html"))]
    seen = set()
    unique_files = []
    for f in html_files:
        if f.name not in seen:
            seen.add(f.name)
            unique_files.append(f)

    if not unique_files:
        print("No HTML files found in", script_dir)
        return

    all_rows = []
    fieldnames = ["source_file", "month", "year", "record_type", "day"] + DATA_COLUMNS[1:]

    for fp in unique_files:
        print(f"Parsing {fp.name} ...")
        try:
            rows = parse_one_file(fp)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  Error: {e}")
            raise

    all_rows = [r for r in all_rows if r.get("record_type") == "daily"]

    out_path = script_dir / "precipitation_consolidated.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} daily rows to {out_path}")


if __name__ == "__main__":
    main()
