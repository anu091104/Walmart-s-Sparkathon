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
from datetime import datetime, timedelta
import random

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
    .category-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin-bottom: 1rem;
    }
    .decision-order {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .decision-reduce {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .decision-hold {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuration parameters with category-specific settings
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


def generate_sample_data(num_products=10, days=90):
    """Generate sample retail inventory data"""
    np.random.seed(42)  # For reproducible results

    categories = list(CATEGORY_PARAMS.keys())
    product_ids = [f"P{str(i).zfill(3)}" for i in range(1, num_products + 1)]

    data = []

    for pid in product_ids:
        category = random.choice(categories)
        base_price = random.uniform(10, 100)
        base_demand = random.uniform(5, 25)
        inventory_level = random.randint(50, 200)

        # Generate time series data
        start_date = datetime.now() - timedelta(days=days)

        for day in range(days):
            current_date = start_date + timedelta(days=day)

            # Add seasonality and trend
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day / 30)  # Monthly seasonality
            trend_factor = 1 + 0.001 * day  # Slight upward trend

            # Price variation (±10%)
            price_variation = random.uniform(0.9, 1.1)
            current_price = base_price * price_variation

            # Demand influenced by price (price elasticity)
            price_elasticity = -0.5
            demand_from_price = base_demand * (current_price / base_price) ** price_elasticity

            # Add randomness and seasonality
            units_sold = max(0, int(demand_from_price * seasonal_factor * trend_factor * random.uniform(0.7, 1.3)))

            # Update inventory
            inventory_level = max(0, inventory_level - units_sold)

            # Replenishment logic (simulate restocking)
            if inventory_level < 20:
                inventory_level += random.randint(50, 150)

            data.append({
                'Date': current_date.strftime('%Y-%m-%d'),
                'Product ID': pid,
                'Category': category,
                'Inventory Level': inventory_level,
                'Units Sold': units_sold,
                'Price': round(current_price, 2)
            })

    return pd.DataFrame(data)


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
            st.warning("Some negative inventory values found - converting to 0")
            df['Inventory Level'] = df['Inventory Level'].clip(lower=0)

        return df

    def process_product(self, product_df, pid):
        """Process individual product data"""
        try:
            product_data = {'Product ID': pid}
            latest_inventory = product_df.iloc[-1]['Inventory Level']
            product_data['Current Inventory'] = latest_inventory

            # Get category and category-specific parameters
            if 'Category' in product_df.columns:
                category = product_df.iloc[-1]['Category']
                product_data['Category'] = category
                params = CATEGORY_PARAMS.get(category, {
                    'ORDERING_COST': 50,
                    'HOLDING_COST': 2.0,
                    'LEAD_TIME_DAYS': 7
                })
                ordering_cost = params['ORDERING_COST']
                holding_cost = params['HOLDING_COST']
                lead_time_days = params['LEAD_TIME_DAYS']
            else:
                product_data['Category'] = 'Unknown'
                ordering_cost = 50
                holding_cost = 2.0
                lead_time_days = 7

            product_data['Ordering Cost'] = ordering_cost
            product_data['Holding Cost'] = holding_cost
            product_data['Lead Time Days'] = lead_time_days

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

            # Inventory calculations using category-specific parameters
            safety_stock = max(MIN_SAFETY_STOCK, round(Z * forecast_std * np.sqrt(lead_time_days)))
            reorder_point = round(avg_daily_demand * lead_time_days + safety_stock)
            eoq = max(1, round(np.sqrt((2 * avg_daily_demand * 365 * ordering_cost) / holding_cost)))

            # Simulate inventory
            inventory_sim = self.simulate_inventory(forecast_30, latest_inventory, reorder_point, eoq)
            stockout_days = sum(1 for inv in inventory_sim if inv <= 0)

            # Decision logic
            if latest_inventory < reorder_point:
                decision = f"ORDER {max(eoq, reorder_point - latest_inventory)} units"
                decision_type = "ORDER"
            elif latest_inventory > 1.5 * reorder_point:
                decision = f"REDUCE by {latest_inventory - reorder_point} units"
                decision_type = "REDUCE"
            else:
                decision = "HOLD current level"
                decision_type = "HOLD"

            product_data.update({
                'Avg Daily Demand': round(avg_daily_demand, 1),
                'Safety Stock': safety_stock,
                'Reorder Point': reorder_point,
                'EOQ': eoq,
                'Stockout Days': stockout_days,
                'Decision': decision,
                'Decision Type': decision_type,
                'Forecast': forecast,
                'Inventory Simulation': inventory_sim,
                'Model': model,
                'Prophet Data': prophet_df
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

        # Choose best price (highest sales, then lowest price in case of tie)
        best_price = max(test_prices, key=lambda p: (suggestions[p], -p))
        best_sales = suggestions[best_price]

        # Calculate revenue optimization
        price_range = np.linspace(current_price * 0.5, current_price * 1.5, 100)
        predicted_sales = price_model.predict(price_range.reshape(-1, 1))
        revenue = price_range * predicted_sales
        max_revenue_idx = np.argmax(revenue)
        optimal_price = price_range[max_revenue_idx]

        return {
            'Current Price': current_price,
            'Suggested Price': round(best_price, 2),
            'Predicted Sales at Best Price': round(best_sales, 2),
            'Optimal Price for Revenue': round(optimal_price, 2),
            'Price Model': price_model,
            'Pricing Data': pricing_df
        }

    def simulate_inventory(self, forecast_30, initial_inventory, reorder_point, eoq):
        """Simulate inventory levels"""
        inventory = []
        current_stock = initial_inventory

        for _, row in forecast_30.iterrows():
            demand = max(0, row['yhat'])  # Ensure non-negative demand
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
    prophet_data = product_data['Prophet Data']

    fig = go.Figure()

    # Historical actual data
    fig.add_trace(go.Scatter(
        x=prophet_data['ds'],
        y=prophet_data['y'],
        mode='markers',
        name='Historical Data',
        marker=dict(color='blue', size=6)
    ))

    # Historical forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'][:-30],
        y=forecast['yhat'][:-30],
        mode='lines',
        name='Historical Forecast',
        line=dict(color='blue', width=2)
    ))

    # Future forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'][-30:],
        y=forecast['yhat'][-30:],
        mode='lines',
        name='Future Forecast',
        line=dict(color='red', dash='dash', width=2)
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
        template='plotly_white',
        height=400
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
        mode='lines+markers',
        name='Projected Inventory',
        line=dict(color='green', width=3),
        marker=dict(size=4)
    ))

    # Reorder point
    fig.add_hline(
        y=product_data['Reorder Point'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Reorder Point ({product_data['Reorder Point']})",
        annotation_position="bottom right"
    )

    # Safety stock
    fig.add_hline(
        y=product_data['Safety Stock'],
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Safety Stock ({product_data['Safety Stock']})",
        annotation_position="top right"
    )

    # Stockout zones
    for i, inv in enumerate(inventory_sim):
        if inv <= 0:
            fig.add_vrect(
                x0=forecast['ds'].iloc[-30 + i],
                x1=forecast['ds'].iloc[-30 + i + 1] if i < len(inventory_sim) - 1 else forecast['ds'].iloc[-1],
                fillcolor="red",
                opacity=0.3,
                line_width=0,
                annotation_text="Stockout" if i == 0 else ""
            )

    fig.update_layout(
        title=f"Inventory Projection - {product_data['Decision']}",
        xaxis_title="Date",
        yaxis_title="Inventory Level",
        hovermode='x unified',
        template='plotly_white',
        height=400
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
    optimal_price = product_data['Optimal Price for Revenue']

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
        subplot_titles=('Price vs Sales Relationship', 'Price vs Revenue Optimization'),
        vertical_spacing=0.15
    )

    # Price vs Sales
    fig.add_trace(
        go.Scatter(
            x=price_range,
            y=predicted_sales,
            mode='lines',
            name='Predicted Sales',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    # Historical price-sales points
    fig.add_trace(
        go.Scatter(
            x=pricing_df['Price'],
            y=pricing_df['Units Sold'],
            mode='markers',
            name='Historical Data',
            marker=dict(color='lightblue', size=6)
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
            line=dict(color='purple', width=2)
        ),
        row=2, col=1
    )

    # Max revenue point
    max_revenue_idx = np.argmax(revenue)
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


def create_category_analysis(results):
    """Create category-wise analysis"""
    category_data = {}

    for result in results:
        category = result.get('Category', 'Unknown')
        if category not in category_data:
            category_data[category] = {
                'products': [],
                'total_inventory': 0,
                'total_demand': 0,
                'order_decisions': 0,
                'stockout_days': 0
            }

        category_data[category]['products'].append(result['Product ID'])
        category_data[category]['total_inventory'] += result['Current Inventory']
        category_data[category]['total_demand'] += result['Avg Daily Demand']
        category_data[category]['stockout_days'] += result['Stockout Days']

        if result['Decision Type'] == 'ORDER':
            category_data[category]['order_decisions'] += 1

    return category_data


def main():
    st.markdown('<h1 class="main-header">📊 Advanced Retail Inventory Management Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Configuration")

    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["Auto-generate Sample Data", "Upload CSV File"]
    )

    if data_source == "Auto-generate Sample Data":
        st.sidebar.markdown("### Sample Data Parameters")
        num_products = st.sidebar.slider("Number of Products", 5, 20, 10)
        num_days = st.sidebar.slider("Days of Historical Data", 30, 180, 90)

        # Generate sample data
        df = generate_sample_data(num_products, num_days)
        st.sidebar.success(f"✅ Generated {len(df)} records for {num_products} products")

    else:
        # File upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload your retail_store_inventory.csv file"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success("✅ File uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {str(e)}")
                return
        else:
            st.info("👈 Please upload a CSV file or use auto-generated sample data")
            return

    try:
        # Initialize analyzer and validate data
        analyzer = InventoryAnalyzer()
        df = analyzer.validate_data(df)

        # Display data summary
        st.subheader("📋 Data Overview")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Products", df['Product ID'].nunique())
        with col2:
            st.metric("Total Records", len(df))
        with col3:
            st.metric("Date Range", f"{(df['Date'].max() - df['Date'].min()).days} days")
        with col4:
            st.metric("Avg Inventory", f"{df['Inventory Level'].mean():.1f}")
        with col5:
            st.metric("Total Categories", df['Category'].nunique() if 'Category' in df.columns else 0)

        # Category distribution
        if 'Category' in df.columns:
            st.subheader("🏷️ Category Distribution")
            category_counts = df['Category'].value_counts()

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Products by Category"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Category Parameters")
                for category, params in CATEGORY_PARAMS.items():
                    if category in category_counts.index:
                        st.markdown(f"""
                        **{category}**
                        - Ordering Cost: ${params['ORDERING_COST']}
                        - Holding Cost: ${params['HOLDING_COST']}
                        - Lead Time: {params['LEAD_TIME_DAYS']} days
                        """)

        # Product selection
        st.subheader("🔍 Product Selection")

        col1, col2 = st.columns(2)

        with col1:
            if 'Category' in df.columns:
                selected_categories = st.multiselect(
                    "Filter by Category",
                    options=df['Category'].unique(),
                    default=df['Category'].unique()
                )
                available_products = df[df['Category'].isin(selected_categories)]['Product ID'].unique()
            else:
                available_products = df['Product ID'].unique()

        with col2:
            selected_products = st.multiselect(
                "Select Products to Analyze",
                options=available_products,
                default=available_products[:5] if len(available_products) > 5 else available_products
            )

        if selected_products:
            # Process selected products
            st.subheader("⚡ Processing Products...")
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, pid in enumerate(selected_products):
                status_text.text(f"Processing product {pid}...")
                product_df = df[df['Product ID'] == pid].copy()
                result = analyzer.process_product(product_df, pid)

                if result:
                    results.append(result)

                progress_bar.progress((i + 1) / len(selected_products))

            status_text.text("✅ Processing complete!")

            if results:
                # Category Analysis
                if 'Category' in df.columns:
                    st.subheader("📊 Category Analysis")
                    category_analysis = create_category_analysis(results)

                    cols = st.columns(len(category_analysis))
                    for i, (category, data) in enumerate(category_analysis.items()):
                        with cols[i]:
                            st.markdown(f"""
                            <div class="category-card">
                                <h3>{category}</h3>
                                <p><strong>Products:</strong> {len(data['products'])}</p>
                                <p><strong>Total Inventory:</strong> {data['total_inventory']}</p>
                                <p><strong>Avg Daily Demand:</strong> {data['total_demand']:.1f}</p>
                                <p><strong>Order Decisions:</strong> {data['order_decisions']}</p>
                                <p><strong>Stockout Days:</strong> {data['stockout_days']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                # Summary Dashboard
                st.subheader("📈 Executive Summary")

                # Key metrics
                col1, col2, col3, col4 = st.columns(4)

                order_count = sum(1 for r in results if r['Decision Type'] == 'ORDER')
                reduce_count = sum(1 for r in results if r['Decision Type'] == 'REDUCE')
                total_stockout_days = sum(r['Stockout Days'] for r in results)
                avg_inventory_level = sum(r['Current Inventory'] for r in results) / len(results)

                with col1:
                    st.metric("Products Needing Orders", order_count, delta=f"{order_count}/{len(results)}")
                with col2:
                    st.metric("Overstock Products", reduce_count, delta=f"{reduce_count}/{len(results)}")
                with col3:
                    st.metric("Total Stockout Days", total_stockout_days)
                with col4:
                    st.metric("Avg Inventory Level", f"{avg_inventory_level:.1f}")

                # Summary table
                st.subheader("📋 Summary Table")
                summary_data = []
                for result in results:
                    summary_data.append({
                        'Product ID': result['Product ID'],
                        'Category': result.get('Category', 'Unknown'),
                        'Current Inventory': result['Current Inventory'],
                        'Reorder Point': result['Reorder Point'],
                        'Safety Stock': result['Safety Stock'],
                        'EOQ': result['EOQ'],
                        'Avg Daily Demand': result['Avg Daily Demand'],
                        'Stockout Days': result['Stockout Days'],
                        'Decision': result['Decision'],
                        'Current Price': result.get('Current Price', 'N/A'),
                        'Suggested Price': result.get('Suggested Price', 'N/A'),
                        'Lead Time': result.get('Lead Time Days', 'N/A')
                    })

                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)

                # Individual product analysis
                st.subheader("🔍 Detailed Product Analysis")

                for result in results:
                    with st.expander(
                            f"📦 Product {result['Product ID']} - {result.get('Category', 'Unknown')} Category"):
                        # Key metrics
                        col1, col2, col3, col4, col5 = st.columns(5)

                        with col3:
                            st.metric("Safety Stock", result['Safety Stock'])
                        with col4:
                            st.metric("EOQ", result['EOQ'])
                        with col5:
                            st.metric("Stockout Days", result['Stockout Days'])

                        # Inventory Decision
                        if result['Decision Type'] == 'ORDER':
                            st.markdown(
                                f"<div class='decision-order'><strong>Decision:</strong> {result['Decision']}</div>",
                                unsafe_allow_html=True)
                        elif result['Decision Type'] == 'REDUCE':
                            st.markdown(
                                f"<div class='decision-reduce'><strong>Decision:</strong> {result['Decision']}</div>",
                                unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f"<div class='decision-hold'><strong>Decision:</strong> {result['Decision']}</div>",
                                unsafe_allow_html=True)

                        # Demand Forecast Plot
                        st.markdown("##### Demand Forecast")
                        fig1 = create_demand_forecast_plot(result)
                        st.plotly_chart(fig1, use_container_width=True)

                        # Inventory Projection Plot
                        st.markdown("##### Inventory Projection")
                        fig2 = create_inventory_projection_plot(result)
                        st.plotly_chart(fig2, use_container_width=True)

                        # Pricing Analysis (if available)
                        if 'Price Model' in result:
                            st.markdown("##### Pricing Analysis")
                            fig3 = create_pricing_analysis_plot(result)
                            if fig3:
                                st.plotly_chart(fig3, use_container_width=True)
                            st.markdown(f"""
                                    - **Current Price:** ${result['Current Price']}
                                    - **Suggested Price:** ${result['Suggested Price']}
                                    - **Optimal Price for Revenue:** ${result['Optimal Price for Revenue']}
                                    - **Predicted Sales at Best Price:** {result['Predicted Sales at Best Price']}
                                """)
                        else:
                            st.info("No pricing analysis available for this product.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
