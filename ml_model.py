def analyze_trends(transactions):
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import numpy as np

def analyze_trends(transactions):
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import numpy as np

    df = pd.DataFrame(transactions)

    required_columns = ["amount", "transaction_date", "type"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Faltam colunas: {', '.join(required_columns)}")

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["year_month"] = df["transaction_date"].dt.to_period("M").astype(str)

    trends = (
        df.groupby(["year_month", "type"])["amount"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    trends["year_month"] = pd.to_datetime(trends["year_month"])
    trends["timestamp"] = trends["year_month"].map(lambda x: x.timestamp())

    predictions = {}
    for transaction_type in ["gain", "expense"]:
        if transaction_type in trends:
            X = trends[["timestamp"]].values
            y = trends[transaction_type].values

            model = LinearRegression()
            model.fit(X, y)

            future_dates = [
                trends["timestamp"].max() + i * 30 * 24 * 3600 for i in range(1, 4)
            ]
            future_predictions = model.predict(np.array(future_dates).reshape(-1, 1))

            # Garantir que os valores sejam sempre positivos
            future_predictions = np.abs(future_predictions)
            predictions[transaction_type] = list(future_predictions)

    return {
        "current_trends": trends.to_dict(orient="records"),
        "predictions": predictions
    }