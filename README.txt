Removal Efficiency Predictor Interface

Open index.html in a browser to use the interface.

This version mirrors the heat-pipe predictor UI and uses:
- Feature names from PW_FINAL_Notebook.ipynb
- Notebook validation metrics, where ETR is the best model
- 500 rows from Data points.xlsx exported into data.js
- A browser-side standardized weighted nearest-neighbor estimator

Important:
The app now uses the supplied dataset directly. To make the interface match the notebook's exact Extra Trees predictions, export the fitted sklearn pipeline as .joblib/.pkl or convert it to a browser/API-backed prediction service.
