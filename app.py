import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Restaurant AI Manager",
    page_icon="🍕",
    layout="wide"
)

st.title("🍕 Restaurant Demand & Smart Inventory AI")
st.write(
    "Forecast food demand, monitor ingredient stock, "
    "and identify purchasing priorities."
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    return joblib.load(
        os.path.join(PROJECT_DIR, "restaurant_demand_models.pkl")
    )

models = load_models()

st.sidebar.header("Forecast Settings")

forecast_date = st.sidebar.date_input(
    "Forecast date",
    value=pd.Timestamp.today().date()
)

is_holiday = st.sidebar.selectbox(
    "Is it a holiday?",
    ["No", "Yes"]
)

promotion = st.sidebar.selectbox(
    "Will there be a promotion?",
    ["No", "Yes"]
)

if st.button("Generate Demand Forecast"):
    day_of_week = pd.Timestamp(forecast_date).dayofweek

    input_data = pd.DataFrame([{
        "DayOfWeek": day_of_week,
        "IsWeekend": int(day_of_week >= 5),
        "Month": forecast_date.month,
        "IsHoliday": int(is_holiday == "Yes"),
        "Promotion": int(promotion == "Yes")
    }])

    forecast = {}

    for food, model in models.items():
        forecast[food] = max(
            0, int(round(model.predict(input_data)[0]))
        )

    forecast_df = pd.DataFrame({
        "Food Item": list(forecast.keys()),
        "Predicted Demand": list(forecast.values())
    })

    st.subheader("📈 Predicted Demand")
    st.dataframe(forecast_df, use_container_width=True)

    st.bar_chart(
        forecast_df.set_index("Food Item")["Predicted Demand"]
    )

    st.download_button(
        "Download Forecast CSV",
        forecast_df.to_csv(index=False),
        file_name="demand_forecast.csv",
        mime="text/csv"
    )

st.caption(
    "Educational prototype. Predictions use a synthetic training dataset "
    "and should be validated with real restaurant sales before operational use."
)
