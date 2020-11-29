"""This script is used to test the Flask app to predict.
It will read the weatherAUS.csv file and will call the Flask app /predict method.
"""
import pandas as pd
import requests

#FLASK_SERVICE_URL = "http://localhost:9696/predict"
FLASK_SERVICE_URL = "https://manning-rain-predict.herokuapp.com/predict"

def main():
    """Read Weather file, select records call flask app"""
    # Load Rain in Australia Dataset
    df_weather_csv = pd.read_csv("data/weatherAUS.csv")
    # Remove RISN_NM
    # Drop variable RISK_NM as it contains information about whether it will rain or not
    df_weather_csv.drop(columns=["RISK_MM"], inplace=True)
    # Extract month, year, day
    df_weather_csv['month'] = pd.DatetimeIndex(df_weather_csv['Date']).month
    df_weather_csv['year'] = pd.DatetimeIndex(df_weather_csv['Date']).year
    df_weather_csv['day'] = pd.DatetimeIndex(df_weather_csv['Date']).day
    # Drop original Date column
    df_weather_csv.drop(columns="Date", inplace=True)
    # Assign target variable as a category type - RainTomorrow
    df_weather_csv['RainToday'] = df_weather_csv['RainToday'].astype('category')
    # Convert to 1 or 0 using cat.codes
    df_weather_csv['RainToday'] = df_weather_csv['RainToday'].cat.codes

    # Drop independent variable since it will be predicted
    df_weather_csv.drop(columns="RainTomorrow", inplace=True)

    # Randomly select unique records for each categorical column combination.
    df_sample = df_weather_csv.sample(142193).drop_duplicates(subset=['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'], keep='first')

    json_data = df_sample.to_json(orient="records")
    # Call Flask App
    results = requests.post(FLASK_SERVICE_URL, json=json_data)
    # Print results
    print(results.text)

if __name__ == "__main__":
    main()
