# S_d(r)_calculation
code for Sd(r) calculation
# S_d(r) Calculation App

This is a Streamlit web app to calculate **S_d(r)** from LAMMPS trajectory files (single frame, format: id type x y z).

## Features

- Upload a LAMMPS dump file
- Specify `R_MIN`, `R_MAX`, and number of bins
- Calculate Sd(r) and vector sums
- Optional further analysis:
  - Upload atom frequency file to classify atoms into liquid-like or solid-like
  - Use XGBoost model to predict frequencies and plot curves
- Download results


## Notes

- Maximum file size for upload: ~200MB
- Large systems may take a while to compute
- ML model prediction requires `xgboost.model` in the repository
