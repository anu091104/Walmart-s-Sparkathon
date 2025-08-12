import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="📊 Inventory & Pricing Forecast", layout="wide")

# Configuration
CATEGORY_PARAMS = {
    'Groceries': {'ORDERING_COST': 30, 'HOLDING_COST': 1.5, 'LEAD_TIME_DAYS': 2},
    'Toys': {'ORDERING_COST': 40, 'HOLDING_COST': 2.0, 'LEAD_TIME_DAYS': 5},
    'Electronics': {'ORDERING_COST': 60, 'HOLDING_COST': 3.5, 'LEAD_TIME_DAYS': 10},
    'Furniture': {'ORDERING_COST': 100, 'HOLDING_COST': 5.0, 'LEAD_TIME_DAYS': 14},
    'Clothing': {'ORDERING_COST': 50, 'HOLDING_COST': 2.5, 'LEAD_TIME_DAYS': 7},
}
SERVICE_LEVEL = 0.95
Z = norm.ppf(SERVICE_LEVEL)
MIN_SAFETY_STOCK = 1
PRICE_TEST_RANGE = 5
MIN_DATA_POINTS = 5

# Generate Sample Data
def generate_data(num_products=5, days=90):
    products = [f"P{str(i+1).zfill(3)}" for i in range(num_products)]
    categories = list(CATEGORY_PARAMS.keys())
    data = []
    for pid in products:
        category = random.choice(categories)
        price_base = random.uniform(20, 100)
        demand_base = random.randint(10, 50)
        inventory = random.randint(50, 150)
        for day in range(days):
            date = datetime.today() - timedelta(days=(days - day))
            seasonality = 1 + 0.3 * np.sin(2 * np.pi * day / 30)
            trend = 1 + 0.001 * day
            price = price_base * random.uniform(0.9, 1.1)
            elasticity = -0.5
            demand_price_effect = demand_base * (price / price_base) ** elasticity
            units_sold = max(0, int(demand_price_effect * seasonality * trend * random.uniform(0.7, 1.3)))
            inventory = max(0, inventory - units_sold)
            if inventory < 20:
                inventory += random.randint(30, 80)
            data.append({
                'Date': date,
                'Product ID': pid,
                'Category': category,
                'Inventory Level': inventory,
                'Units Sold': units_sold,
                'Price': round(price, 2)
            })
    return pd.DataFrame(data)

# Data Generation
df = generate_data(num_products=10, days=120)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(['Product ID', 'Date'], inplace=True)

# Product Selection
st.title("📦 Inventory Forecasting & Price Optimization")
product_ids = df['Product ID'].unique()
selected = st.multiselect("Select Products", product_ids, default=product_ids[:3])

if not selected:
    st.warning("Please select at least one product to view analysis.")

for pid in selected:
    product_df = df[df['Product ID'] == pid].copy()
    st.header(f"🔍 Product {pid}")
    latest_inventory = product_df.iloc[-1]['Inventory Level']
    category = product_df['Category'].iloc[-1]
    params = CATEGORY_PARAMS[category]

    prophet_df = product_df[['Date', 'Units Sold']].rename(columns={'Date': 'ds', 'Units Sold': 'y'})
    prophet_df['price'] = product_df['Price']
    model = Prophet(seasonality_mode='multiplicative')
    model.add_regressor('price')
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=30)
    future['price'] = prophet_df['price'].mean()
    forecast = model.predict(future)
    forecast_30 = forecast.tail(30)

    avg_demand = forecast_30['yhat'].mean()
    std = (prophet_df['y'] - model.predict(prophet_df)['yhat']).std()
    std = 0 if np.isnan(std) else std
    safety_stock = max(MIN_SAFETY_STOCK, round(Z * std * np.sqrt(params['LEAD_TIME_DAYS'])))
    reorder_point = round(avg_demand * params['LEAD_TIME_DAYS'] + safety_stock)
    eoq = max(1, round(np.sqrt((2 * avg_demand * 365 * params['ORDERING_COST']) / params['HOLDING_COST'])))

    # Inventory Simulation
    sim_inventory = []
    current_stock = latest_inventory
    stockout_days = 0
    for _, row in forecast_30.iterrows():
        demand = row['yhat']
        current_stock -= demand
        if current_stock < 0:
            stockout_days += 1
            current_stock = 0
        if current_stock <= reorder_point:
            current_stock += eoq
        sim_inventory.append(current_stock)

    # Price Optimization
    price_model = LinearRegression()
    pricing_df = product_df[['Price', 'Units Sold']].dropna()
    if len(pricing_df) >= 4:
        price_model.fit(pricing_df[['Price']], pricing_df['Units Sold'])
        current_price = pricing_df['Price'].iloc[-1]
        test_prices = np.linspace(current_price * 0.8, current_price * 1.2, 100)
        predicted_sales = price_model.predict(test_prices.reshape(-1, 1))
        revenue = test_prices * predicted_sales
        best_idx = np.argmax(revenue)
        suggested_price = test_prices[best_idx]

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=test_prices, y=predicted_sales, name="Predicted Sales", line=dict(color='blue')))
        fig_price.add_trace(go.Scatter(x=test_prices, y=revenue, name="Revenue", yaxis="y2", line=dict(color='green')))
        fig_price.update_layout(
            title="📈 Price Optimization",
            xaxis=dict(title="Price ($)"),
            yaxis=dict(title="Units Sold", color="blue"),
            yaxis2=dict(title="Revenue ($)", color="green", overlaying="y", side="right"),
            height=400
        )
        st.plotly_chart(fig_price, use_container_width=True)
        st.success(f"💡 Suggested Price: ${suggested_price:.2f}")
    else:
        st.warning("Not enough data for pricing analysis.")

    # Forecast Plot
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='markers', name='Actual'))
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig_forecast.update_layout(title="📊 Demand Forecast", height=400)
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Inventory Projection
    fig_inventory = go.Figure()
    fig_inventory.add_trace(go.Scatter(x=forecast_30['ds'], y=sim_inventory, mode='lines+markers', name='Inventory'))
    fig_inventory.add_hline(y=reorder_point, line_dash="dash", line_color="red", annotation_text="Reorder Point")
    fig_inventory.add_hline(y=safety_stock, line_dash="dot", line_color="orange", annotation_text="Safety Stock")
    fig_inventory.update_layout(title="📦 Inventory Projection", height=400)
    st.plotly_chart(fig_inventory, use_container_width=True)

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("EOQ", eoq)
    col2.metric("Stockout Days", stockout_days)
    col3.metric("Reorder Point", reorder_point)
    st.divider()
