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
</style>
""", unsafe_allow_html=True)

# Configuration parameters
SERVICE_LEVEL = 0.95
Z = norm.ppf(SERVICE_LEVEL)
ORDERING_COST = 50
HOLDING_COST = 2
MIN_SAFETY_STOCK = 1
LEAD_TIME_DAYS = 7
PRICE_TEST_RANGE = 5
MIN_DATA_POINTS = 5


class InventoryAnalyzer:
    def __init__(self):
        self.results = []

    def validate_data(self, df):
        """Validate uploaded data"""
        required_cols = {'Date', 'Product ID', 'Inventory Level', 'Units Sold'}
        missing_cols = required_cols - set(df.columns)

        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Product ID', 'Date'])

        if df['Inventory Level'].min() < 0:
            raise ValueError("Negative inventory values found")

        return df

    def process_product(self, product_df, pid):
        """Process individual product data"""
        try:
            product_data = {'Product ID': pid}
            latest_inventory = product_df.iloc[-1]['Inventory Level']
            product_data['Current Inventory'] = latest_inventory

            if len(product_df) < 7:
                return None

            # Prepare Prophet data
            prophet_df = product_df[['Date', 'Units Sold']].rename(
                columns={'Date': 'ds', 'Units Sold': 'y'}
            )

            # Add price regressor if available
            if 'Price' in product_df.columns:
                prophet_df['price'] = product_df['Price']

            # Initialize and fit Prophet model
            model = Prophet(
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                daily_seasonality=False,
                yearly_seasonality=True
            )

            if 'price' in prophet_df.columns:
                model.add_regressor('price')

            model.fit(prophet_df)

            # Create future dataframe
            future = model.make_future_dataframe(periods=30)

            if 'price' in prophet_df.columns:
                future['price'] = prophet_df['price'].mean()

            forecast = model.predict(future)
            forecast_30 = forecast.tail(30)

            # Calculate demand metrics
            avg_daily_demand = forecast_30['yhat'].mean()
            forecast_std = (prophet_df['y'] - model.predict(prophet_df)['yhat']).std()

            # Dynamic Pricing Analysis
            pricing_results = self.analyze_pricing(product_df, pid)
            if pricing_results:
                product_data.update(pricing_results)

            # Inventory calculations
            safety_stock = max(MIN_SAFETY_STOCK, round(Z * forecast_std * np.sqrt(LEAD_TIME_DAYS)))
            reorder_point = round(avg_daily_demand * LEAD_TIME_DAYS + safety_stock)
            eoq = max(1, round(np.sqrt((2 * avg_daily_demand * 365 * ORDERING_COST) / HOLDING_COST)))

            # Simulate inventory
            inventory_sim = self.simulate_inventory(forecast_30, latest_inventory, reorder_point, eoq)
            stockout_days = sum(1 for inv in inventory_sim if inv <= 0)

            # Decision logic
            if latest_inventory < reorder_point:
                decision = f"ORDER {max(eoq, reorder_point - latest_inventory)} units"
            elif latest_inventory > 1.5 * reorder_point:
                decision = f"REDUCE by {latest_inventory - reorder_point} units"
            else:
                decision = "HOLD current level"

            product_data.update({
                'Avg Daily Demand': round(avg_daily_demand, 1),
                'Safety Stock': safety_stock,
                'Reorder Point': reorder_point,
                'EOQ': eoq,
                'Stockout Days': stockout_days,
                'Decision': decision,
                'Forecast': forecast,
                'Inventory Simulation': inventory_sim,
                'Model': model
            })

            return product_data

        except Exception as e:
            st.error(f"Error processing product {pid}: {str(e)}")
            return None

    def analyze_pricing(self, product_df, pid):
        """Analyze pricing for a product"""
        if 'Price' not in product_df.columns or len(product_df) < MIN_DATA_POINTS:
            return None

        pricing_df = product_df.dropna(subset=['Price', 'Units Sold'])

        if len(pricing_df) <= 3:
            return None

        X = pricing_df[['Price']]
        y = pricing_df['Units Sold']

        price_model = LinearRegression()
        price_model.fit(X, y)

        current_price = pricing_df['Price'].iloc[-1]

        # Generate test prices
        test_prices = [
            max(0, current_price - PRICE_TEST_RANGE),
            current_price,
            max(0, current_price + PRICE_TEST_RANGE)
        ]

        # Predict sales for each test price
        suggestions = {}
        for test_price in test_prices:
            pred = price_model.predict([[test_price]])[0]
            suggestions[test_price] = max(0, pred)

        # Choose best price
        best_price = max(test_prices, key=lambda p: (suggestions[p], -p))
        best_sales = suggestions[best_price]

        return {
            'Current Price': current_price,
            'Suggested Price': round(best_price, 2),
            'Predicted Sales at Best Price': round(best_sales, 2),
            'Price Model': price_model,
            'Pricing Data': pricing_df
        }

    def simulate_inventory(self, forecast_30, initial_inventory, reorder_point, eoq):
        """Simulate inventory levels"""
        inventory = []
        current_stock = initial_inventory

        for _, row in forecast_30.iterrows():
            demand = row['yhat']
            current_stock -= demand

            if current_stock < 0:
                current_stock = 0

            # Replenishment logic
            if current_stock <= reorder_point:
                current_stock += eoq

            inventory.append(current_stock)

        return inventory


def create_demand_forecast_plot(product_data):
    """Create demand forecast plot using Plotly"""
    forecast = product_data['Forecast']
    model = product_data['Model']

    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=forecast['ds'][:-30],
        y=forecast['yhat'][:-30],
        mode='lines',
        name='Historical Forecast',
        line=dict(color='blue')
    ))

    # Future forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'][-30:],
        y=forecast['yhat'][-30:],
        mode='lines',
        name='Future Forecast',
        line=dict(color='red', dash='dash')
    ))

    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'][-30:],
        y=forecast['yhat_upper'][-30:],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'][-30:],
        y=forecast['yhat_lower'][-30:],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        name='Confidence Interval',
        fillcolor='rgba(255,0,0,0.2)'
    ))

    fig.update_layout(
        title=f"Demand Forecast for Product {product_data['Product ID']}",
        xaxis_title="Date",
        yaxis_title="Units Sold",
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def create_inventory_projection_plot(product_data):
    """Create inventory projection plot"""
    forecast = product_data['Forecast']
    inventory_sim = product_data['Inventory Simulation']

    fig = go.Figure()

    # Inventory projection
    fig.add_trace(go.Scatter(
        x=forecast['ds'][-30:],
        y=inventory_sim,
        mode='lines',
        name='Projected Inventory',
        line=dict(color='green', width=3)
    ))

    # Reorder point
    fig.add_hline(
        y=product_data['Reorder Point'],
        line_dash="dash",
        line_color="red",
        annotation_text="Reorder Point"
    )

    # Safety stock
    fig.add_hline(
        y=product_data['Safety Stock'],
        line_dash="dot",
        line_color="orange",
        annotation_text="Safety Stock"
    )

    # Stockout zones
    for i, inv in enumerate(inventory_sim):
        if inv <= 0:
            fig.add_vrect(
                x0=forecast['ds'].iloc[-30 + i],
                x1=forecast['ds'].iloc[-30 + i + 1] if i < len(inventory_sim) - 1 else forecast['ds'].iloc[-1],
                fillcolor="red",
                opacity=0.3,
                line_width=0
            )

    fig.update_layout(
        title=f"Inventory Projection - {product_data['Decision']}",
        xaxis_title="Date",
        yaxis_title="Inventory Level",
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def create_pricing_analysis_plot(product_data):
    """Create pricing analysis plot"""
    if 'Price Model' not in product_data:
        return None

    price_model = product_data['Price Model']
    pricing_df = product_data['Pricing Data']
    current_price = product_data['Current Price']
    suggested_price = product_data['Suggested Price']

    # Generate price range for visualization
    price_range = np.linspace(
        pricing_df['Price'].min() * 0.8,
        pricing_df['Price'].max() * 1.2,
        100
    )

    predicted_sales = price_model.predict(price_range.reshape(-1, 1))
    revenue = price_range * predicted_sales

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price vs Sales', 'Price vs Revenue'),
        vertical_spacing=0.1
    )

    # Price vs Sales
    fig.add_trace(
        go.Scatter(
            x=price_range,
            y=predicted_sales,
            mode='lines',
            name='Predicted Sales',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    # Current price point
    fig.add_trace(
        go.Scatter(
            x=[current_price],
            y=[price_model.predict([[current_price]])[0]],
            mode='markers',
            name=f'Current Price: ${current_price:.2f}',
            marker=dict(color='red', size=12)
        ),
        row=1, col=1
    )

    # Suggested price point
    fig.add_trace(
        go.Scatter(
            x=[suggested_price],
            y=[price_model.predict([[suggested_price]])[0]],
            mode='markers',
            name=f'Suggested Price: ${suggested_price:.2f}',
            marker=dict(color='green', size=12, symbol='star')
        ),
        row=1, col=1
    )

    # Revenue curve
    fig.add_trace(
        go.Scatter(
            x=price_range,
            y=revenue,
            mode='lines',
            name='Revenue',
            line=dict(color='purple')
        ),
        row=2, col=1
    )

    # Max revenue point
    max_revenue_idx = np.argmax(revenue)
    optimal_price = price_range[max_revenue_idx]

    fig.add_trace(
        go.Scatter(
            x=[optimal_price],
            y=[revenue[max_revenue_idx]],
            mode='markers',
            name=f'Max Revenue: ${optimal_price:.2f}',
            marker=dict(color='purple', size=12, symbol='diamond')
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=f"Price Analysis for Product {product_data['Product ID']}",
        height=600,
        template='plotly_white',
        showlegend=True
    )

    fig.update_xaxes(title_text="Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Price ($)", row=2, col=1)
    fig.update_yaxes(title_text="Predicted Sales", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($)", row=2, col=1)

    return fig


def main():
    st.markdown('<h1 class="main-header">📊 Retail Inventory Management Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.markdown("## Configuration")

    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload your retail_store_inventory.csv file"
    )

    if uploaded_file is not None:
        try:
            # Load and validate data
            df = pd.read_csv(uploaded_file)
            analyzer = InventoryAnalyzer()
            df = analyzer.validate_data(df)

            st.sidebar.success("✅ Data validated successfully!")

            # Display data summary
            st.subheader("📋 Data Summary")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Products", df['Product ID'].nunique())
            with col2:
                st.metric("Total Records", len(df))
            with col3:
                st.metric("Date Range",
                          f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
            with col4:
                st.metric("Avg Inventory", f"{df['Inventory Level'].mean():.1f}")

            # Product selection
            selected_products = st.sidebar.multiselect(
                "Select Products to Analyze",
                options=df['Product ID'].unique(),
                default=df['Product ID'].unique()[:3] if len(df['Product ID'].unique()) > 3 else df[
                    'Product ID'].unique()
            )

            if selected_products:
                st.subheader("🔍 Analysis Results")

                # Process selected products
                results = []
                progress_bar = st.progress(0)

                for i, pid in enumerate(selected_products):
                    product_df = df[df['Product ID'] == pid].copy()
                    result = analyzer.process_product(product_df, pid)

                    if result:
                        results.append(result)

                    progress_bar.progress((i + 1) / len(selected_products))

                if results:
                    # Summary table
                    summary_data = []
                    for result in results:
                        summary_data.append({
                            'Product ID': result['Product ID'],
                            'Current Inventory': result['Current Inventory'],
                            'Reorder Point': result['Reorder Point'],
                            'Safety Stock': result['Safety Stock'],
                            'EOQ': result['EOQ'],
                            'Avg Daily Demand': result['Avg Daily Demand'],
                            'Stockout Days': result['Stockout Days'],
                            'Decision': result['Decision'],
                            'Suggested Price': result.get('Suggested Price', 'N/A'),
                            'Current Price': result.get('Current Price', 'N/A')
                        })

                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)

                    # Individual product analysis
                    for result in results:
                        st.subheader(f"📦 Product {result['Product ID']} - Detailed Analysis")

                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Current Inventory", result['Current Inventory'])
                        with col2:
                            st.metric("Reorder Point", result['Reorder Point'])
                        with col3:
                            st.metric("Safety Stock", result['Safety Stock'])
                        with col4:
                            st.metric("EOQ", result['EOQ'])

                        # Decision alert
                        if "ORDER" in result['Decision']:
                            st.error(f"🚨 {result['Decision']}")
                        elif "REDUCE" in result['Decision']:
                            st.warning(f"⚠️ {result['Decision']}")
                        else:
                            st.success(f"✅ {result['Decision']}")

                        # Pricing information
                        if 'Suggested Price' in result and result['Suggested Price'] != 'N/A':
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Current Price", f"${result['Current Price']:.2f}")
                            with col2:
                                st.metric("Suggested Price", f"${result['Suggested Price']:.2f}")

                        # Visualizations
                        col1, col2 = st.columns(2)

                        with col1:
                            # Demand forecast
                            fig1 = create_demand_forecast_plot(result)
                            st.plotly_chart(fig1, use_container_width=True)

                        with col2:
                            # Inventory projection
                            fig2 = create_inventory_projection_plot(result)
                            st.plotly_chart(fig2, use_container_width=True)

                        # Pricing analysis (if available)
                        if 'Price Model' in result:
                            fig3 = create_pricing_analysis_plot(result)
                            if fig3:
                                st.plotly_chart(fig3, use_container_width=True)

                        st.markdown("---")

                    # Download results
                    csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Analysis Results",
                        data=csv,
                        file_name="inventory_analysis_results.csv",
                        mime="text/csv"
                    )

                else:
                    st.warning("No products could be analyzed. Please check your data.")

            else:
                st.info("Please select at least one product to analyze.")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    else:
        st.info("👈 Please upload a CSV file to begin analysis")

        # Sample data format
        st.subheader("📄 Required Data Format")
        sample_data = {
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Product ID': ['P001', 'P001', 'P002'],
            'Inventory Level': [100, 95, 80],
            'Units Sold': [5, 8, 12],
            'Price': [25.00, 25.00, 30.00]
        }

        st.dataframe(pd.DataFrame(sample_data))
        st.caption("Note: Price column is optional but recommended for pricing analysis")


if __name__ == "__main__":
    main()