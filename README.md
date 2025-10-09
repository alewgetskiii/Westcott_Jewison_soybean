# Understanding and extending the WAOB Crop Weather Model for Soybeans 

This project reproduces and extends the **USDA World Agricultural Outlook Board (WAOB)** crop weather model for **U.S. soybean yield forecasting**, as originally described by **Westcott & Jewison (2013)** and later updated by **Irwin & Hubbs (2020)**.

The goal is to **replicate**, **analyze**, and **update** the regression-based model used by the USDA to predict national and state-level soybean yields based on weather and trend variables.

---

## Project Structure

```
project-root/
│
├── data_fetch_nat_weather_harvest.py     # Script for downloading and preparing NOAA & USDA datasets
├── main_woab_model.ipynb                 # Core modeling, regression, and yield forecast analysis
├── conditions_feature_explore.ipynb      # Feature engineering and exploratory analysis of crop conditions
├── fdd090720.pdf                         # Reference paper: Farmdoc Daily (Irwin & Hubbs, 2020)
├── WAOB_Report.pdf                       # Final research report
│
├── data/
│   ├── raw/                              # Raw downloaded datasets
│   ├── processed/                        # Cleaned and model-ready datasets
│   └── results/                          # Outputs: coefficients, forecasts, and metrics
│
├── src/
│   └── fetch_crops_condition.py          # Script for downloading crop condition data
│
├── .venv/                                # Local virtual environment (optional if using Conda)
├── environment-m1.yml                    # Conda environment file
└── README.md                             # Project documentation

```

---

## Environment Setup

Before running any code, you need to create and activate the project environment
From the root of the repository:

```bash
# Create the environment
conda env create -f environment-m1.yml

# Activate it
conda activate waob-m1
```

This environment includes the main Python packages used for data science:
`numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `plotly`, `seaborn`, and `scikit-learn`.

---

## How to Run

1. **Download the data:**

   * Open `data_fetch_nat_weather_harvest.py`
   * Provide your USDA Quicstats API key [Link](https://quickstats.nass.usda.gov/api)
   * Run all cells to fetch and prepare NOAA Climate Division and USDA QuickStats data

2. **Run the main model:**

   * Open `main.ipynb`
   * This notebook:

     * Builds the regression model following Westcott–Jewison (2013)
     * Generates updated national and state-level yield forecasts
     * Saves outputs to `data/processed/`

3. **Outputs:**

   * Metrics and coefficients for baseline and augmented models are saved as CSVs:

     ```
     data/results/waob_outputs_all.csv
     ```

---

## References

* **Westcott, P.C. & Jewison, M. (2013)**
  *Weather Effects on Expected Soybean and Soybean Yields.*
  USDA Economic Research Service, FDS-13g-01.
  [Link](https://www.ers.usda.gov/publications/pub-details/?pubid=36652)

* **Irwin, S. & Hubbs, T. (2020)**
  *Understanding the WAOB Crop Weather Model for Soybeans.*
  farmdoc daily (10):126, Department of Agricultural and Consumer Economics, University of Illinois.
  [Link](https://farmdocdaily.illinois.edu/2020/07/understanding-the-waob-crop-weather-model-for-soybeans.html)

* **NOAA Climate Division Data Documentation:**
  [Link](https://www.ncei.noaa.gov/pub/data/cirs/climdiv/state-readme.txt)

