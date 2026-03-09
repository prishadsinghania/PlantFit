# PlantFit: Precision Irrigation Management

*We propose a novel data-driven crop management system designed to optimize irrigation schedules using real-time weather data, ET API, sap flow levels and predictive modeling.*
This project integrates atmospheric forecasts with plant-specific metrics to ensure efficient water usage and optimal growth for popcorn crops.

**Problem Statement:** Crops are thirsty and we often know too late. This causes great loss in productivity. We integrate a minimum invasive crop sap flow sensor to see the “pulse of the crop” and combine it with environmental data and forecasts to provide irrigation advice and answer questions like "when should i water the plant" and "how much should the plant be watered".

**Team Member**

Prisha Singhania - pds4@illinois.edu, UIUC, (MSIM’26 Grad Student)

Priyal Maniar - priyalm2@illinois.edu, UIUC, (MSIM’26 Grad Student)

Yuyang Liu - yuyang19@illinois.edu, UIUC, (MSIM’27 Grad Student)

Ximin Pian - xpiao2@illinois.edu, UIUC, (PhD candidate in CEE)

## 🚀 Key Features

- **Automated Irrigation Modeling:** Execute core agricultural logic via `run_irrigation_model.py` to calculate precise watering needs.
    
- **Weather Integration:** Processes 48-hour forecasts and historical precipitation data to adjust for upcoming environmental changes.
    
- **Zone Management:** Modular control for different field sectors, allowing for specialized treatment based on soil type or crop stage.
    
- **Baseline Calibration:** Utilizes a `trained_baseline.csv` to compare current field data against historical performance standards.
    
- **Visual Prototyping:** Includes an interactive interface (`plantfit_prototype.html`) for monitoring system status and plant health.

- **Data Pipeline:** An NSFI-driven pipeline using satellite and sensor data to trigger real-time plant stress alerts.
![WhatsApp Image 2026-03-08 at 11 30 52](https://github.com/user-attachments/assets/e42a88eb-29e7-48b0-a4cf-febf000625bc)

---

## 📂 Repository Structure

|**File / Folder**|**Function**|
|---|---|
|`precipitation/`|Scripts for parsing and analyzing rainfall events.|
|`sap/`|Modules handling Soil-Plant-Atmosphere (SAP) sensor data.|
|`weather/`|API integrations for localized weather monitoring and forecasting.|
|`zone management/`|Logic for subdividing and managing specific agricultural plots.|
|`run_irrigation_model.py`|The main execution script that synthesizes data into recommendations.|
|`trained_baseline.csv`|Pre-calculated data used to calibrate the model's predictions.|
|`mock_48hr_forecast.csv`|Sample weather data for testing the model without live API calls.|
|`requirements.txt`|List of Python dependencies required for the system.|

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
    
- Pip (Python package manager)
    

### Installation

1. **Clone the repository:**
    
    Bash
    
    ```
    git clone https://github.com/prishadsinghania/PlantFit
    cd illini_popcorn
    ```
    
2. **Install dependencies:**
    
    Bash
    
    ```
    pip install -r requirements.txt
    ```

3. **External API**
    API key was originally stored in the .env file but have also include it in the api_key file

---

## 📈 Usage

The system is designed to be modular, allowing for both automated modeling and manual data adjustment.

### Running the Irrigation Model

To generate irrigation recommendations based on current forecasts and crop baselines, run the primary script:

Bash

```
python run_irrigation_model.py
```

### Data Customization

- **Forecast Adjustments:** You can manually update `mock_48hr_forecast.csv` with specific temperature or humidity data to simulate different environmental stress tests.
    
- **Model Calibration:** The `trained_baseline.csv` file serves as the ground truth for "healthy" crop behavior. Updating this allows the model to adapt to different corn hybrids.
    

### App Dashboard Prototype

To view the conceptual UI for this system, open `plantfit_prototype.html` in any web browser. This dashboard provides a visual representation of soil moisture levels and plant health trends.
(<img width="600" height="1300" alt="image" src="https://github.com/user-attachments/assets/2b6f70ac-6613-42d0-b289-57bba05215c6" />
)


---

## 📄 License

This project is licensed under the **MIT License**
