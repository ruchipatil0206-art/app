import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

st.set_page_config(page_title="Stock Forecasting App", layout="wide")

st.title("📈 Stock Price Forecasting using ARIMA")

ticker = st.text_input(
    "Enter Stock Ticker",
    value="RELIANCE.NS"
)

if st.button("Run Forecast"):

    try:
        # Last 5 years data
        end_date = datetime.today()
        start_date = pd.DateOffset(years=5)

        data = yf.download(
            ticker,
            start=end_date - start_date,
            end=end_date,
            auto_adjust=True
        )

        if data.empty:
            st.error("No data found.")
            st.stop()

        close = data["Close"]

        st.subheader("Historical Price Data")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=close.index,
                y=close,
                mode="lines",
                name="Close Price"
            )
        )

        fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(fig, use_container_width=True)

        # --------------------
        # ARIMA MODEL
        # --------------------
        st.subheader("ARIMA Forecast")

        model = ARIMA(close, order=(5, 1, 0))
        model_fit = model.fit()

        # Forecast until June 2027
        target_date = pd.Timestamp("2027-06-30")

        days_to_forecast = (
            target_date - close.index[-1]
        ).days

        forecast = model_fit.forecast(
            steps=max(days_to_forecast, 1)
        )

        future_dates = pd.date_range(
            start=close.index[-1] + pd.Timedelta(days=1),
            periods=len(forecast),
            freq="D"
        )

        forecast_df = pd.DataFrame(
            {
                "Date": future_dates,
                "Forecast": forecast
            }
        )

        june_2027 = forecast_df[
            forecast_df["Date"] >= "2027-06-01"
        ]

        predicted_price = float(
            june_2027.iloc[-1]["Forecast"]
        )

        st.metric(
            "Predicted Price (June 2027)",
            f"{predicted_price:.2f}"
        )

        forecast_fig = go.Figure()

        forecast_fig.add_trace(
            go.Scatter(
                x=close.index,
                y=close,
                name="Historical"
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=forecast,
                name="Forecast"
            )
        )

        forecast_fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )

        st.subheader("Forecast Sample")

        st.dataframe(
            forecast_df.tail(20),
            use_container_width=True
        )

    except Exception as e:
        st.error(str(e))
