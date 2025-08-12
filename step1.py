import pandas as pd

# Load dataset
df = pd.read_csv("C:/Users/Anuska/Downloads/retail_store_inventory.csv")


# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract useful date features
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday
df['Year'] = df['Date'].dt.year

# Check for null values
missing = df.isnull().sum()

# Display unique categories for quick checks
categories = {
    'Weather': df['Weather Condition'].unique(),
    'Seasonality': df['Seasonality'].unique(),
    'Regions': df['Region'].unique(),
    'Categories': df['Category'].unique()
}

# Preview the cleaned dataset
print("Missing values:\n", missing)
print("\nUnique values:\n", categories)
print("\nSample data:\n", df.head())
