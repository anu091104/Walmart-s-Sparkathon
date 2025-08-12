import pandas as pd

# Load your main dataset
df = pd.read_csv("C:/Users/Anuska/Downloads/retail_store_inventory.csv")

# New products to add
new_products = pd.DataFrame({
    'Date': ['2023-07-01'],
    'Store ID': ['S005'],
    'Product ID': ['P0999'],
    'Category': ['Beauty'],
    'Region': ['West'],
    'Inventory Level': [100],
    'Units Sold': [40],
    'Units Ordered': [60],
    'Demand Forecast': [70],
    'Price': [20.5],
    'Discount': [5],
    'Weather Condition': ['Sunny'],
    'Holiday/Promotion': ['None'],
    'Competitor Pricing': [19.5],
    'Seasonality': ['Summer'],
    'Month': [7],
    'Day': [1],
    'Weekday': [5],
    'Year': [2023]
})

# Append to main dataset
df = pd.concat([df, new_products], ignore_index=True)

# Save it back (optional)
df.to_csv("updated_inventory.csv", index=False)
