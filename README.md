# Student Academic Performance Prediction

## Run locally
1. Extract this project folder.
2. Open a terminal in the folder.
3. Create and activate a virtual environment.
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `streamlit run app.py`

## Deploy on Streamlit Community Cloud
1. Upload the contents of this folder to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the main file path to `app.py`.

## Important model note
The supplied model is the Linear Regression model from the uploaded Colab bundle.
Its recorded test metrics are MAE 3.740, RMSE 4.573, and R² -0.020.
The negative R² means it performed slightly worse than predicting the test-set mean
on that particular split. Predictions are estimates only.

The feature-analysis CSV contains absolute regression coefficient magnitudes, not
normalized feature-importance percentages.
