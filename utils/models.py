from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
import pandas as pd
import numpy as np


def train_salary_model(df):
    """
    Trains a Linear Regression model to predict salary based on study hours.

    Args:
        df: DataFrame containing 'horas_estudo_mes' and 'salario' columns.

    Returns:
        dict: A dictionary containing the trained model, test data, and metrics.
    """
    # Prepare data
    X = np.array(df["horas_estudo_mes"]).reshape(-1, 1)
    y = df["salario"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    r2_score = model.score(X_test, y_test)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "r2_score": r2_score,
        "X_all": X,
        "y_all": y,
    }


def predict_salary(model, hours):
    """
    Predicts salary for a given number of study hours.

    Args:
        model: Trained LinearRegression model.
        hours: Number of hours to predict salary for.

    Returns:
        float: Predicted salary.
    """
    input_data = np.array([[hours]])
    prediction = model.predict(input_data)
    return prediction[0]


def train_rent_model(df):
    """
    Trains a Linear Regression model to predict rent based on area.

    Args:
        df: DataFrame containing 'area_m2' and 'valor_aluguel' columns.

    Returns:
        dict: A dictionary containing the trained model, test data, and metrics.
    """
    # Prepare data
    X = np.array(df["area_m2"]).reshape(-1, 1)
    y = df["valor_aluguel"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    r2_score = model.score(X_test, y_test)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "r2_score": r2_score,
        "X_all": X,
        "y_all": y,
    }


def predict_rent(model, area):
    """
    Predicts rent for a given area.

    Args:
        model: Trained LinearRegression model.
        area: Area in m2 to predict rent for.

    Returns:
        float: Predicted rent.
    """
    input_data = np.array([[area]])
    prediction = model.predict(input_data)
    return prediction[0]


def train_sales_model(df):
    """
    Trains a Simple Exponential Smoothing model for sales prediction.

    Args:
        df: DataFrame containing 'Data' and 'Total_Vendas' columns.

    Returns:
        dict: A dictionary containing the trained model and the original series.
    """
    # Preprocessing
    df_proc = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_proc["Data"]):
        df_proc["Data"] = pd.to_datetime(df_proc["Data"])

    series = df_proc.set_index("Data")["Total_Vendas"]
    series = series.asfreq("D")  # Ensure daily frequency

    # Model Training (Simple Exponential Smoothing)
    # Using smoothing_level=0.2 as per original notebook
    model = SimpleExpSmoothing(series)
    fitted_model = model.fit(smoothing_level=0.2, optimized=False)

    return {"model": fitted_model, "series": series}


def evaluate_sales_model(model_data):
    """
    Calculates evaluation metrics (RMSE, MAPE) for the sales model.

    Args:
        model_data: Dictionary containing the trained model and series.

    Returns:
        dict: Dictionary with RMSE and MAPE.
    """
    model = model_data["model"]
    series = model_data["series"]

    # In-sample predictions (fitted values)
    predictions = model.fittedvalues
    actuals = series

    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    return {"rmse": rmse, "mape": mape}


def predict_sales(model_data, days=31):
    """
    Forecasts sales for a specified number of days.

    Args:
        model_data: The dictionary returned by train_sales_model.
        days: Number of days to forecast.

    Returns:
        pandas.Series: Forecasted values.
    """
    model = model_data["model"]
    forecast = model.forecast(days)
    return forecast


def train_credit_model(df):
    """
    Trains a Random Forest Classifier for credit score prediction.

    Args:
        df: DataFrame containing customer data.

    Returns:
        dict: A dictionary containing the trained model, encoders, and metrics.
    """
    # Preprocessing
    df_proc = df.copy()

    # Drop irrelevant columns if they exist
    cols_to_drop = ["id_cliente", "mes"]  # identifiers
    df_proc = df_proc.drop(columns=[c for c in cols_to_drop if c in df_proc.columns])

    # Encode categorical columns
    encoders = {}
    for col in df_proc.columns:
        if df_proc[col].dtype == "object":
            le = LabelEncoder()
            df_proc[col] = le.fit_transform(df_proc[col])
            encoders[col] = le

    # Split data
    X = df_proc.drop("score_credito", axis=1)
    y = df_proc["score_credito"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model (Random Forest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Feature Importance
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return {
        "model": model,
        "encoders": encoders,
        "accuracy": accuracy,
        "feature_importance": feature_importance,
        "X_test": X_test,
        "y_test": y_test,
    }


def predict_credit(model_data, input_data):
    """
    Predicts credit score for a given input.

    Args:
        model_data: Dictionary containing the trained model and encoders.
        input_data: Dictionary or DataFrame with input features.

    Returns:
        str: Predicted credit score (decoded).
    """
    model = model_data["model"]
    encoders = model_data["encoders"]

    # Create DataFrame if input is dict -> ensures column order if dict keys match training cols
    if isinstance(input_data, dict):
        # We need to make sure we include all columns expected by the model
        pass  # Handle in page logic to match columns, or here?
        # Better to expect a proper dataframe or dict matching schema.
        # Let's assume input_df will be constructed in the page.

    input_df = (
        pd.DataFrame([input_data])
        if isinstance(input_data, dict)
        else input_data.copy()
    )

    # Preprocess Input using saved encoders
    for col, le in encoders.items():
        if col in input_df.columns:
            # Handle unseen labels carefully or just try transform
            try:
                input_df[col] = le.transform(input_df[col])
            except ValueError:
                # If unseen label, maybe assign a default or handled error?
                # For now, let's assume valid inputs from UI
                # Fallback: simple fit_transform on a single item won't work for inference
                # We'll rely on UI restricting choices to known categories
                pass

    # Filter columns to align with model training features
    if hasattr(model, "feature_names_in_"):
        input_df = input_df[model.feature_names_in_]

    prediction_encoded = model.predict(input_df)

    # Decode prediction
    if "score_credito" in encoders:
        prediction = encoders["score_credito"].inverse_transform(prediction_encoded)
        return prediction[0]

    return prediction_encoded[0]


def train_flight_model(df):
    """
    Trains a Random Forest Regressor to predict flight delays.

    Args:
        df: DataFrame containing flight data.

    Returns:
        dict: Trained model, encoders, and metrics.
    """
    # Preprocessing
    df_proc = df.copy()

    # Select features
    features = [
        "AIRLINE_Description",
        "ORIGIN_CITY",
        "DEST_CITY",
        "DISTANCE",
        "DAY_OF_WEEK",
        "TIME_HOUR",
    ]
    target = "DELAY_OVERALL"

    # Drop rows with missing values
    df_proc = df_proc.dropna(subset=features + [target])

    # Filter for numeric delay
    df_proc = df_proc[df_proc[target] >= 0]

    # Encode categorical columns
    encoders = {}
    for col in ["AIRLINE_Description", "ORIGIN_CITY", "DEST_CITY", "DAY_OF_WEEK"]:
        le = LabelEncoder()
        # Convert to string to ensure consistent encoding
        df_proc[col] = df_proc[col].astype(str)
        df_proc[col] = le.fit_transform(df_proc[col])
        encoders[col] = le

    X = df_proc[features]
    y = df_proc[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    # Using limited depth and estimators for interactivity speed
    model = RandomForestRegressor(
        n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    return {
        "model": model,
        "encoders": encoders,
        "mae": mae,
        "X_train_columns": features,
    }


def predict_flight_delay(model_data, input_data):
    """
    Predicts delay for a given flight input.

    Args:
        model_data: Dictionary containing the trained model and encoders.
        input_data: Dictionary with input features.

    Returns:
        float: Predicted delay in minutes.
    """
    model = model_data["model"]
    encoders = model_data["encoders"]
    features = model_data["X_train_columns"]

    # Prepare input dataframe
    input_df = pd.DataFrame([input_data])

    # Apply encoders
    for col, le in encoders.items():
        if col in input_df.columns:
            val = str(input_df.iloc[0][col])
            # Handle unknown labels gracefully (assign to most frequent or 0)
            if val in le.classes_:
                input_df[col] = le.transform([val])
            else:
                # If unknown, use the first class code (often 0)
                input_df[col] = 0

    # Ensure correct column order
    input_df = input_df[features]

    prediction = model.predict(input_df)
    return max(0, prediction[0])  # Delay shouldn't be negative
