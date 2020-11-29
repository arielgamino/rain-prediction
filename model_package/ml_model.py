import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from statistics import mode

# Load trained model
log_regression = load('log_regression_project_1.joblib')

def clean_data(df_to_clean):
    """Take a dataframe, clean and transform it"""
    # Columns with missing data
    df_values_missing = df_to_clean.isna().sum()
    df_values_missing = df_values_missing[df_values_missing>0]
    # Retrieve columns that need to be imputed
    columns_missing = df_values_missing.index
    # Select numerical and categorical columns (based on those with missing data)
    numerical_columns_missing   = list(df_to_clean[columns_missing]._get_numeric_data().columns)
    categorical_columns_missing = list(set(df_to_clean[columns_missing].columns) - set(numerical_columns_missing))

    # For each column with missing values in training, impute
    # Use median to impute to prevent outliers from skewing the imputed value
    for column in numerical_columns_missing:
        median_value= df_to_clean[column].median()
        df_to_clean[column].fillna(median_value, inplace=True)

    # Categorical values
    # For each column with missing values in training, impute
    # Use mode to impute since it's a categorical variable
    for column in categorical_columns_missing:
        mode_value= mode(df_to_clean[column].dropna())
        df_to_clean[column].fillna(mode_value, inplace=True)

    # For missing values we will use the most frequent (0)
    df_to_clean['RainToday'] = df_to_clean['RainToday'].apply(lambda rain_today: rain_today if(rain_today>=0) else 0)
    # Columns with ouliers (as determined in Milestone 3)
    outlier_columns = ['Rainfall','Evaporation','WindGustSpeed','WindSpeed9am','WindSpeed3pm']
    quantile_values = [.9998,.998, .99, .998, .99]

    # For each of the outlier variables, cap the max values
    for index in range(len(outlier_columns)):
        column = outlier_columns[index]
        quantile_value = quantile_values[index]
        cap_value = df_to_clean[column].quantile(quantile_value)
        # Cap the value in training data
        df_to_clean[column] = df_to_clean[column].apply(lambda value: value if(value<=cap_value) else cap_value)

    # Get name of numerical and categorical columns
    numerical_columns = list(df_to_clean._get_numeric_data().columns)
    categorical_columns = list(set(df_to_clean.columns) - set(numerical_columns))

    # Get dummies and concatenate to current dataframe, drop original categorical_columns
    df_data = pd.concat([df_to_clean,pd.get_dummies(df_to_clean[categorical_columns])], axis=1)
    df_data.drop(columns=categorical_columns, inplace=True)

    # At this point all columns are numeric

    # Shrink range of data by putting within a range of 0 and 1
    scaler = MinMaxScaler()
    # Scale training data
    data_np = scaler.fit_transform(df_data)
    df_data = pd.DataFrame(data_np, columns=df_data.columns)

    return df_data

def predict_rain(json_data):
    """Given list of records predict whether it will rain or not"""
    # Convert json_data to dataframe
    df_json_data = pd.DataFrame(json_data)
    # Clean the data
    df_cleaned = clean_data(df_json_data)
    # Make rain predictions
    prediction = log_regression.predict(df_cleaned)
    predictions = []
    probabilities = []
    predicted_probabilities = log_regression.predict_proba(df_cleaned)
    # Retrieve probabilities depending on whether is a 0 (no rain) or 1 (rain)
    for index in range(len(predicted_probabilities)):
        predicted = prediction[index]
        # Convert to Yes or No
        predicted_str = "Yes" if predicted == 1 else "No"
        # Based on prediction retrieve probability
        probability = predicted_probabilities[index][predicted]
        predictions.append(predicted_str)
        probabilities.append(probability)

    return {'predictions':predictions,'probability':probabilities}