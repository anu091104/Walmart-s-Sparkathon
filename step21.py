import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/Anuska/Downloads/retail_store_inventory.csv")
df['Date'] = pd.to_datetime(df['Date'])

# 🔧 Add this line to fix the error:
df['Month'] = df['Date'].dt.month
# 1. Monthly Sales Trend
plt.figure(figsize=(10, 6))
monthly_sales = df.groupby('Month')['Units Sold'].sum()
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker='o')
plt.title('📅 Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Units Sold')
plt.xticks(range(1, 13))  # Ensure months 1–12
plt.grid(True)
plt.show()

# 2. Category-wise Sales
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Category', y='Units Sold', estimator=sum)
plt.title('📦 Category-wise Sales')
plt.ylabel('Total Units Sold')
plt.xticks(rotation=20)
plt.show()

# 3. Sales by Weather
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Weather Condition', y='Units Sold')
plt.title('🌤 Sales vs Weather Condition')
plt.show()

# 4. Sales by Season
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Seasonality', y='Units Sold', estimator=sum)
plt.title('🍁 Seasonality vs Sales')
plt.xticks(rotation=20)
plt.show()