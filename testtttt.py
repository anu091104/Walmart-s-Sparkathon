import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import warnings
from tkinter import Tk, filedialog

warnings.filterwarnings("ignore")


# File selection dialog
def select_file():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    return file_path


# STEP 2: Select CSV and Validate Data
print("Please select 'retail_store_inventory.csv' (Columns: Date, Product ID, Inventory Level, Units Sold)")
file_path = select_file()

if not file_path:
    raise FileNotFoundError("No file selected")

# Load and validate data
df = pd.read_csv(file_path)
assert {'Date', 'Product ID', 'Inventory Level', 'Units Sold'}.issubset(df.columns), "Missing required columns"
df['Date'] = pd.to_datetime(df['Date'])
assert df['Date'].is_monotonic_increasing, "Data must be sorted by date"

# CONFIGURATION
SERVICE_LEVEL = 0.95  # 95% confidence
Z = norm.ppf(SERVICE_LEVEL)  # z-score
ORDERING_COST = 50  # Cost per order ($)
HOLDING_COST = 2  # Annual holding cost per unit ($)

# STEP 3: Enhanced Product Processing
product_ids = df['Product ID'].unique()

for pid in product_ids:
    print(f"\nProcessing Product: {pid}")

    product_df = df[df['Product ID'] == pid].copy()
    latest_inventory = product_df.iloc[-1]['Inventory Level']

    if latest_inventory == 0:
        print("New product detected - requires manual setup")
        continue

    prophet_df = product_df[['Date', 'Units Sold']].rename(columns={'Date': 'ds', 'Units Sold': 'y'})

    if len(prophet_df) < 10:
        print("Insufficient data - using simple moving average")
        avg_daily_demand = prophet_df['y'].mean()
        forecast_std = avg_daily_demand * 0.3

        forecast_30 = pd.DataFrame({
            'ds': pd.date_range(start=prophet_df['ds'].max() + pd.Timedelta(days=1), periods=30),
            'yhat': [avg_daily_demand] * 30,
            'yhat_lower': [avg_daily_demand * 0.85] * 30,
            'yhat_upper': [avg_daily_demand * 1.15] * 30
        })

    else:
        model = Prophet(
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            daily_seasonality=False
        )
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        forecast_30 = forecast.tail(30)

        df_merged = prophet_df.merge(forecast[['ds', 'yhat']], on='ds')
        forecast_error = df_merged['y'] - df_merged['yhat']
        forecast_std = max(forecast_error.std(), 0.1 * prophet_df['y'].mean())

    LEAD_TIME_DAYS = product_df['Lead Time Days'].mode()[0] if 'Lead Time Days' in product_df.columns else 7

    avg_daily_demand = forecast_30['yhat'].mean()
    annual_demand = avg_daily_demand * 365
    eoq = np.sqrt((2 * annual_demand * ORDERING_COST) / HOLDING_COST)

    safety_stock = Z * np.sqrt(
        LEAD_TIME_DAYS * forecast_std ** 2 +
        (avg_daily_demand * 0.1) ** 2
    )

    reorder_point = avg_daily_demand * LEAD_TIME_DAYS + safety_stock

    excess_threshold = 1.5 * reorder_point
    if latest_inventory < reorder_point:
        order_qty = max(eoq, reorder_point - latest_inventory)
        decision = f"Reorder {order_qty:.0f} units"
    elif latest_inventory > excess_threshold:
        excess_units = latest_inventory - reorder_point
        decision = f"Reduce inventory by {excess_units:.0f} units"
    else:
        decision = "Maintain current level"

    # STEP 4: Simulate Inventory Over Time
    inventory_over_time = []
    stockout_days = 0
    current_inventory = latest_inventory

    for _, row in forecast_30.iterrows():
        current_inventory -= row['yhat']
        if current_inventory < 0:
            stockout_days += 1
            current_inventory = 0
        inventory_over_time.append(current_inventory)

    # STEP 5: Visualization
    plt.figure(figsize=(13, 6))

    plt.plot(forecast_30['ds'], forecast_30['yhat'],
             label='Forecasted Demand', color='#1f77b4', linewidth=2)
    plt.fill_between(forecast_30['ds'], forecast_30['yhat_lower'],
                     forecast_30['yhat_upper'], alpha=0.2, color='#1f77b4')

    plt.plot(forecast_30['ds'], inventory_over_time,
             label='Projected Inventory', color='#2ca02c', linewidth=3)

    plt.axhline(reorder_point, color='red', linestyle='--', label='Reorder Point')
    plt.axhline(safety_stock, color='orange', linestyle=':', label='Safety Stock')

    if stockout_days > 0:
        plt.fill_between(forecast_30['ds'], 0,
                         0.05 * max(forecast_30['yhat_upper']),
                         where=np.array(inventory_over_time) <= 0,
                         color='red', alpha=0.3, label=f'Stockout Risk ({stockout_days} days)')

    plt.title(f"Product {pid} - {decision}", pad=20)
    plt.xlabel("Date")
    plt.ylabel("Units")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save and show plot
    plt.savefig(f"{pid}_forecast.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Print Summary
    print(f"Current Inventory: {latest_inventory:.0f}")
    print(f"Avg Daily Demand: {avg_daily_demand:.1f} ± {forecast_std:.1f}")
    print(f"Safety Stock: {safety_stock:.1f} (Lead Time: {LEAD_TIME_DAYS} days)")
    print(f"Reorder Point: {reorder_point:.1f}")
    print(f"Optimal Order Qty (EOQ): {eoq:.0f}")
    print(f"Decision: {decision}")
    if stockout_days > 0:
        print(f"WARNING: {stockout_days} days of stockout predicted")

print("\nAnalysis complete. Plots saved to file.")