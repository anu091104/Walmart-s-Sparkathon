import streamlit as st
import pandas as pd
import plotly.express as px

# Page settings
st.set_page_config(page_title="Retail Inventory Dashboard", layout="wide")

# Load the dataset
df = pd.read_csv("C:/Users/Anuska/Downloads/retail_store_inventory.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Sidebar - Product Selection
st.sidebar.title("🛒 Product Filter")
product_ids = df["Product ID"].unique()
selected_product = st.sidebar.selectbox("🔍 Select a Product ID", product_ids)

# Filter data
product_data = df[df["Product ID"] == selected_product].sort_values("Date")
latest = product_data.iloc[-1]

# Header
st.markdown("<h1 style='color:#f9a825;'>📦 Retail Inventory Forecast Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Analyze inventory trends, forecasted demand, and smart decision-making in real-time.")

# === KPI Cards ===
with st.container():
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📦 Inventory Level", int(latest["Inventory Level"]))
    col2.metric("📈 Forecasted Demand", f"{latest['Demand Forecast']:.2f}")
    col3.metric("🛍️ Units Sold", int(latest["Units Sold"]))
    col4.metric("💰 Price", f"${latest['Price']:.2f}")

# === Inventory Line Chart ===
st.markdown("### 📊 Inventory Level Over Time")
fig_inv = px.line(
    product_data, x="Date", y="Inventory Level",
    title="Inventory Trend", markers=True, color_discrete_sequence=["#f9a825"]
)
fig_inv.update_layout(hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_inv, use_container_width=True)

# === Forecast vs Actual Chart ===
st.markdown("### 📉 Forecast vs Actual Sales")
fig_sales = px.line(
    product_data, x="Date",
    y=["Demand Forecast", "Units Sold"],
    labels={"value": "Units", "variable": "Legend"},
    title="Forecasted Demand vs Actual Sales",
    color_discrete_map={
        "Demand Forecast": "#42a5f5",
        "Units Sold": "#66bb6a"
    }
)
fig_sales.update_layout(hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_sales, use_container_width=True)

# === Smart Insight Box ===
st.markdown("### 📌 Smart Inventory Insight")
avg_demand = product_data["Demand Forecast"].mean()
current_inventory = latest["Inventory Level"]

with st.expander("🧠 AI-based Suggestion"):
    st.write(f"📊 **Average Forecasted Demand:** `{avg_demand:.2f}`")
    st.write(f"📦 **Current Inventory Level:** `{current_inventory}`")

    if current_inventory < avg_demand:
        st.error("⚠️ Your inventory is **below average demand**. Reordering is recommended.")
        st.markdown('<span style="color:#ef5350;font-weight:bold;">🔴 Status: Low Inventory</span>', unsafe_allow_html=True)
    elif current_inventory > avg_demand * 1.5:
        st.warning("⚠️ Inventory might be **overstocked**. Consider slowing down orders.")
        st.markdown('<span style="color:#ffb300;font-weight:bold;">🟠 Status: Overstock</span>', unsafe_allow_html=True)
    else:
        st.success("✅ Inventory is healthy and matches forecast.")
        st.markdown('<span style="color:#66bb6a;font-weight:bold;">🟢 Status: Balanced</span>', unsafe_allow_html=True)

# === Footer ===
st.markdown("---")
st.markdown("<center>Made with ❤️ by Anushka Srivastava for Walmart Sparkathon</center>", unsafe_allow_html=True)
