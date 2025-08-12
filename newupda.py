import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Retail Inventory Management Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .final-recommendations {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .chart-container {
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sample final recommendations data
FINAL_RECOMMENDATIONS = {
    'Product ID': ['P0001', 'P0002', 'P0003', 'P0004', 'P0005', 'P0006', 'P0007', 'P0008', 'P0009', 'P0010',
                   'P0011', 'P0012', 'P0013', 'P0014', 'P0015', 'P0016', 'P0017', 'P0018', 'P0019', 'P0020'],
    'Current Inventory': [300, 253, 98, 158, 372, 491, 59, 409, 319, 308, 156, 495, 389, 328, 408, 96, 313, 278, 374,
                          117],
    'Reorder Point': [998, 501, 2497, 1113, 1080, 2489, 521, 1801, 2420, 1125, 543, 1825, 1817, 1111, 2863, 2469, 1140,
                      1494, 1060, 495],
    'Decision': ['ORDER 1328 units', 'ORDER 1387 units', 'ORDER 2399 units', 'ORDER 1417 units', 'ORDER 1421 units',
                 'ORDER 1998 units', 'ORDER 1397 units', 'ORDER 1392 units', 'ORDER 2101 units', 'ORDER 1442 units',
                 'ORDER 1460 units', 'ORDER 1330 units', 'ORDER 1428 units', 'ORDER 1437 units', 'ORDER 2455 units',
                 'ORDER 2373 units', 'ORDER 1432 units', 'ORDER 1466 units', 'ORDER 1396 units', 'ORDER 1388 units'],
    'Suggested Price': [104.33, 29.96, 19.17, 72.29, 101.72, 66.30, 67.12, 58.42, 82.70, 14.20,
                        81.09, 31.94, 37.05, 61.50, 78.26, 78.73, 77.57, 16.11, 48.14, 83.39],
    'Predicted Sales at Best Price': [142.57, 135.14, 135.20, 135.28, 141.73, 135.96, 137.63, 133.56, 137.17, 139.73,
                                      136.37, 136.62, 134.41, 138.44, 138.39, 141.75, 134.60, 130.24, 136.81, 140.21],
    'Stockout Days': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    'Category': ['Toys', 'Groceries', 'Furniture', 'Toys', 'Toys', 'Furniture', 'Groceries', 'Electronics', 'Furniture',
                 'Toys', 'Groceries', 'Electronics', 'Electronics', 'Toys', 'Furniture', 'Furniture', 'Toys',
                 'Clothing',
                 'Toys', 'Groceries']
}


def create_inventory_status_chart():
    """Create inventory status chart"""
    df = pd.DataFrame(FINAL_RECOMMENDATIONS)

    # Calculate status
    df['Status'] = np.where(
        df['Current Inventory'] < df['Reorder Point'],
        'Below Reorder Point',
        np.where(
            df['Current Inventory'] > 1.5 * df['Reorder Point'],
            'Overstocked',
            'Normal'
        )
    )

    fig = px.pie(
        df,
        names='Status',
        title='Inventory Status Distribution',
        color='Status',
        color_discrete_map={
            'Below Reorder Point': '#EF553B',
            'Normal': '#00CC96',
            'Overstocked': '#636EFA'
        }
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

    return fig


def create_category_analysis_chart():
    """Create category analysis chart"""
    df = pd.DataFrame(FINAL_RECOMMENDATIONS)

    # Group by category
    category_df = df.groupby('Category').agg({
        'Current Inventory': 'mean',
        'Reorder Point': 'mean',
        'Stockout Days': 'sum',
        'Product ID': 'count'
    }).reset_index()

    category_df.rename(columns={'Product ID': 'Product Count'}, inplace=True)

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'bar'}, {'type': 'pie'}]])

    # Inventory vs Reorder Point by Category
    fig.add_trace(
        go.Bar(
            x=category_df['Category'],
            y=category_df['Current Inventory'],
            name='Avg Current Inventory',
            marker_color='#636EFA'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=category_df['Category'],
            y=category_df['Reorder Point'],
            name='Avg Reorder Point',
            marker_color='#EF553B'
        ),
        row=1, col=1
    )

    # Stockout Days by Category
    fig.add_trace(
        go.Pie(
            labels=category_df['Category'],
            values=category_df['Stockout Days'],
            name='Stockout Days',
            hole=0.4
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text='Category Analysis',
        barmode='group',
        showlegend=True
    )

    return fig


def create_price_vs_sales_chart():
    """Create price vs sales chart"""
    df = pd.DataFrame(FINAL_RECOMMENDATIONS)

    fig = px.scatter(
        df,
        x='Suggested Price',
        y='Predicted Sales at Best Price',
        color='Category',
        size='Current Inventory',
        hover_name='Product ID',
        title='Price vs Predicted Sales',
        trendline="lowess"
    )

    fig.update_layout(
        xaxis_title='Suggested Price ($)',
        yaxis_title='Predicted Sales (units)'
    )

    return fig


def create_inventory_vs_reorder_chart():
    """Create inventory vs reorder point chart"""
    df = pd.DataFrame(FINAL_RECOMMENDATIONS)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df['Product ID'],
            y=df['Current Inventory'],
            mode='markers+lines',
            name='Current Inventory',
            line=dict(color='blue', width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df['Product ID'],
            y=df['Reorder Point'],
            mode='markers+lines',
            name='Reorder Point',
            line=dict(color='red', width=2, dash='dash')
        )
    )

    fig.update_layout(
        title='Current Inventory vs Reorder Point by Product',
        xaxis_title='Product ID',
        yaxis_title='Units',
        hovermode='x unified'
    )

    return fig


def main():
    st.markdown('<h1 class="main-header">📊 Retail Inventory Management Dashboard</h1>', unsafe_allow_html=True)

    # Display final recommendations table
    st.subheader("Final Recommendations")
    st.dataframe(pd.DataFrame(FINAL_RECOMMENDATIONS), use_container_width=True)

    # Charts section
    st.subheader("Key Metrics and Visualizations")

    # First row of charts
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_inventory_status_chart(), use_container_width=True)

    with col2:
        st.plotly_chart(create_price_vs_sales_chart(), use_container_width=True)

    # Second row of charts
    st.plotly_chart(create_category_analysis_chart(), use_container_width=True)

    # Third row of charts
    st.plotly_chart(create_inventory_vs_reorder_chart(), use_container_width=True)

    # Add download button for the recommendations
    csv = pd.DataFrame(FINAL_RECOMMENDATIONS).to_csv(index=False)
    st.download_button(
        label="Download Final Recommendations",
        data=csv,
        file_name="inventory_recommendations.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()