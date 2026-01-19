from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
