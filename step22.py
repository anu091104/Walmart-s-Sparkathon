import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load your dataset
file_path = "C:/Users/Anuska/Downloads/retail_store_inventory.csv"
df = pd.read_csv(file_path)

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Group by Date and Category to get daily sales per category
category_data = df.groupby(['Date', 'Category'])['Units Sold'].sum().reset_index()

# Get unique categories
categories = category_data['Category'].unique()

# Forecast for each category
for category in categories:
    print(f"\n🔍 Forecasting for Category: {category}")

    # Filter data for this category
    cat_df = category_data[category_data['Category'] == category][['Date', 'Units Sold']]
    cat_df = cat_df.rename(columns={"Date": "ds", "Units Sold": "y"})

    # Initialize Prophet model
    model = Prophet()
    model.fit(cat_df)

    # Make future dataframe (e.g., 30 days)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Plot the forecast
    plt.figure(figsize=(10, 5))
    model.plot(forecast)
    plt.title(f"Forecasted Units Sold - {category}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.tight_layout()
    plt.show()
