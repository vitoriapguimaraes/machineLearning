from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
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
