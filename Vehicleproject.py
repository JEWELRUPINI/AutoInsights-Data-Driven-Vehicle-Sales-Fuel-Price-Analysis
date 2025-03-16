#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os

# Define file paths (update if needed)
parent_folder = r"C:\Users\joelr\OneDrive\Documents"

# Load datasets
df_tv = pd.read_csv(os.path.join(parent_folder, "traditional_vehicle.csv"))
df_ev = pd.read_csv(os.path.join(parent_folder, "EV.csv"))
df_fuel = pd.read_csv(os.path.join(parent_folder, "fuel.csv"))

### 🛠 1. Check Column Names
print("\n🟢 Traditional Vehicle Columns:", df_tv.columns.tolist())
print("\n🔵 EV Columns:", df_ev.columns.tolist())
print("\n🟠 Fuel Columns:", df_fuel.columns.tolist())

### 🛠 2. Print First Few Rows
print("\n📌 First few rows of Traditional Vehicle Data:")
print(df_tv.head())

print("\n📌 First few rows of EV Data:")
print(df_ev.head())

print("\n📌 First few rows of Fuel Data:")
print(df_fuel.head())

### 🛠 3. Check for Non-Numeric Issues
def check_column_types(df, name):
    print(f"\n📊 Checking Column Data Types for {name}:")
    print(df.dtypes)
    if "Year" in df.columns:
        print("\n🚨 Non-Numeric Values Detected in Year Column:")
        print(df[~df["Year"].astype(str).str.isnumeric()])

check_column_types(df_fuel, "Fuel Data")

# EV and TV may have Year in a different format, so manually inspect


# In[2]:


import pandas as pd
import os

# Define file paths (update if needed)
parent_folder = r"C:\Users\joelr\OneDrive\Documents"

# Load datasets
df_tv = pd.read_csv(os.path.join(parent_folder, "traditional_vehicle.csv"))
df_ev = pd.read_csv(os.path.join(parent_folder, "EV.csv"))
df_fuel = pd.read_csv(os.path.join(parent_folder, "fuel.csv"))

### 🛠 1. Check Column Names
print("\n🟢 Traditional Vehicle Columns:", df_tv.columns.tolist())
print("\n🔵 EV Columns:", df_ev.columns.tolist())
print("\n🟠 Fuel Columns:", df_fuel.columns.tolist())

### 🛠 2. Print First Few Rows
print("\n📌 First few rows of Traditional Vehicle Data:")
print(df_tv.head())

print("\n📌 First few rows of EV Data:")
print(df_ev.head())

print("\n📌 First few rows of Fuel Data:")
print(df_fuel.head())

### 🛠 3. Check for Non-Numeric Issues
def check_column_types(df, name):
    print(f"\n📊 Checking Column Data Types for {name}:")
    print(df.dtypes)
    if "Year" in df.columns:
        print("\n🚨 Non-Numeric Values Detected in Year Column:")
        print(df[~df["Year"].astype(str).str.isnumeric()])

check_column_types(df_fuel, "Fuel Data")

# 🛠 Handle Column Formatting for Numeric Data (Traditional Vehicle Data)
def clean_traditional_vehicle_data(df):
    # Remove commas and convert to numeric
    cols_to_clean = df.columns[1:]  # Skip the 'Category' column
    for col in cols_to_clean:
        df[col] = df[col].replace({',': ''}, regex=True).astype(float, errors='ignore')
    return df

df_tv = clean_traditional_vehicle_data(df_tv)

# 🛠 Rename Columns for EV Data
df_ev.columns = ['Category', '2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', 'Grand Total']

# 🛠 Handle Negative Values in Traditional Vehicle Data (Optional)
def handle_negative_values(df):
    # Replace negative values with zero or drop rows as needed
    df[df.columns[1:]] = df[df.columns[1:]].applymap(lambda x: max(x, 0) if isinstance(x, (int, float)) else x)
    return df

df_tv = handle_negative_values(df_tv)

# 🛠 Check for Missing Values
def check_missing_values(df, name):
    print(f"\n🛠 Checking for Missing Values in {name}:")
    print(df.isnull().sum())

check_missing_values(df_tv, "Traditional Vehicle Data")
check_missing_values(df_ev, "EV Data")
check_missing_values(df_fuel, "Fuel Data")


# In[3]:


import pandas as pd
import os

# Define file paths (update if needed)
parent_folder = r"C:\Users\joelr\OneDrive\Documents"
output_folder = r"C:\Users\joelr\OneDrive\Documents\CleanedData"  # Define the folder where cleaned data will be saved

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load datasets
df_tv = pd.read_csv(os.path.join(parent_folder, "traditional_vehicle.csv"))
df_ev = pd.read_csv(os.path.join(parent_folder, "EV.csv"))
df_fuel = pd.read_csv(os.path.join(parent_folder, "fuel.csv"))

# 🛠 1. Remove commas and convert to numeric
def clean_commas_and_convert(df, columns):
    for column in columns:
        # Remove commas and convert to numeric (coerce errors to NaN)
        df[column] = pd.to_numeric(df[column].replace({',': ''}, regex=True), errors='coerce')
    return df

# Traditional Vehicle: Columns that need comma removal and conversion to numeric
tv_columns = ['2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']
df_tv = clean_commas_and_convert(df_tv, tv_columns)

# 🛠 2. Handle Missing Values Automatically
def handle_missing_values(df):
    # Fill missing values with 0 (can be adjusted to mean/median)
    return df.fillna(0)

df_tv = handle_missing_values(df_tv)
df_ev = handle_missing_values(df_ev)
df_fuel = handle_missing_values(df_fuel)

# 🛠 3. Remove Duplicate Row in EV Data (First row is duplicated in second row)
df_ev = df_ev.drop(index=0).reset_index(drop=True)  # Drop the first row and reset the index

# 🛠 4. Rename EV Columns
df_ev.columns = ['Category', '2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024', 'Grand Total']

# 🛠 5. Handle Negative Values in Traditional Vehicle Data (for specific rows like 'Quadricycles')
def handle_negative_values_in_rows(df, category_column, value_columns):
    for index, row in df.iterrows():
        if row[category_column] == 'Quadricycles':  # Check for Quadricycles row
            for col in value_columns:
                if pd.notna(row[col]):  # Check if the value is not NaN
                    df.at[index, col] = max(row[col], 0)  # Replace negative values with 0
    return df

# Apply to Traditional Vehicle Data
df_tv = handle_negative_values_in_rows(df_tv, 'Category', tv_columns)

# 🛠 6. Save Cleaned Data
def save_cleaned_data(df, filename):
    output_path = os.path.join(output_folder, filename)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

save_cleaned_data(df_tv, "cleaned_traditional_vehicle.csv")
save_cleaned_data(df_ev, "cleaned_EV.csv")
save_cleaned_data(df_fuel, "cleaned_fuel.csv")

print("Data Cleaning Completed and Saved.")


# In[4]:


import pandas as pd

# Define file paths
parent_folder = r"C:\Users\joelr\OneDrive\Documents\CleanedData"

# Reimport Cleaned Data
df_tv = pd.read_csv(os.path.join(parent_folder, "cleaned_traditional_vehicle.csv"))
df_ev = pd.read_csv(os.path.join(parent_folder, "cleaned_EV.csv"))
df_fuel = pd.read_csv(os.path.join(parent_folder, "cleaned_fuel.csv"))

# Display first few rows to verify
print(df_tv.head())
print(df_ev.head())
print(df_fuel.head())


# In[5]:


import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Define file paths (update if needed)
parent_folder = r"C:\Users\joelr\OneDrive\Documents"
output_folder = r"C:\Users\joelr\OneDrive\Documents\CleanedData"  # Define the folder where cleaned data will be saved

# Load cleaned fuel dataset
df_fuel = pd.read_csv(os.path.join(output_folder, "cleaned_fuel.csv"))

# 🛠 1. Models to test
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

# 🛠 2. Evaluate Fuel Price Prediction
def evaluate_model(df, features_columns, target_column):
    features = df[features_columns]
    target = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    best_model = None
    best_mse = float('inf')

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{name} MSE: {mse}")
        if mse < best_mse:
            best_mse = mse
            best_model = model

    return best_model, best_mse

# Evaluate Fuel Price Prediction
fuel_columns = df_fuel.drop(columns=['Year']).columns  # Use all columns except 'Year' for prediction
best_model_fuel, best_mse_fuel = evaluate_model(df_fuel, fuel_columns, 'Average Petrol Price (₹/litre)')
print(f"Best Model for Fuel Price Prediction: {best_model_fuel.__class__.__name__}, MSE: {best_mse_fuel}")

# 🛠 3. Forecast using the best model for Fuel Price
def make_forecast(best_model, df, features_columns):
    features = df[features_columns]
    forecast = best_model.predict(features)
    return forecast

# Forecast for Fuel Price
forecast_fuel = make_forecast(best_model_fuel, df_fuel, fuel_columns)
print(f"Forecast for Fuel Price: {forecast_fuel}")


# In[6]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np

# Prepare the data (ensure 'Year' is numeric)
years = np.array(df_fuel['Year']).reshape(-1, 1)  # Reshape for sklearn
fuel_prices = df_fuel['Average Petrol Price (₹/litre)'].values

# Split data into training and test sets (80% train, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(years, fuel_prices, test_size=0.2, random_state=42)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_reg_model = LinearRegression()
poly_reg_model.fit(X_poly_train, y_train)
y_pred_poly = poly_reg_model.predict(X_poly_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)

# Linear Regression
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)
y_pred_lin = lin_reg_model.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)

# XGBoost Regressor
xg_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xg_model.fit(X_train, y_train)
y_pred_xg = xg_model.predict(X_test)
mse_xg = mean_squared_error(y_test, y_pred_xg)

# Display MSE for each model
print(f"Linear Regression MSE: {mse_lin}")
print(f"Polynomial Regression MSE: {mse_poly}")
print(f"Random Forest MSE: {mse_rf}")
print(f"Gradient Boosting MSE: {mse_gb}")
print(f"XGBoost MSE: {mse_xg}")

# Determine the best model
mse_values = {
    "Linear Regression": mse_lin,
    "Polynomial Regression": mse_poly,
    "Random Forest": mse_rf,
    "Gradient Boosting": mse_gb,
    "XGBoost": mse_xg
}

best_model_name = min(mse_values, key=mse_values.get)
best_mse = mse_values[best_model_name]

print(f"\nBest Model for Fuel Price Prediction: {best_model_name}, MSE: {best_mse}")

# Forecast for the best model
if best_model_name == "Linear Regression":
    forecast = lin_reg_model.predict([[2024], [2025], [2026], [2027], [2028]])
elif best_model_name == "Polynomial Regression":
    forecast = poly_reg_model.predict(poly.transform([[2024], [2025], [2026], [2027], [2028]]))
elif best_model_name == "Random Forest":
    forecast = rf_model.predict([[2024], [2025], [2026], [2027], [2028]])
elif best_model_name == "Gradient Boosting":
    forecast = gb_model.predict([[2024], [2025], [2026], [2027], [2028]])
else:
    forecast = xg_model.predict([[2024], [2025], [2026], [2027], [2028]])

print(f"Forecast for Future Fuel Prices: {forecast}")


# In[7]:


import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Define file paths (update if needed)
parent_folder = r"C:\Users\joelr\OneDrive\Documents"
output_folder = r"C:\Users\joelr\OneDrive\Documents\CleanedData"

# Load cleaned Traditional Vehicle dataset
df_tv = pd.read_csv(os.path.join(output_folder, "cleaned_traditional_vehicle.csv"))

# 🛠 1. Models to test
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

# 🛠 2. Evaluate Traditional Vehicle Sales Prediction
def evaluate_model(df, features_columns, target_column):
    features = df[features_columns]
    target = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    best_model = None
    best_mse = float('inf')

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{name} MSE: {mse}")
        if mse < best_mse:
            best_mse = mse
            best_model = model

    return best_model, best_mse

# Evaluate Traditional Vehicle Sales Prediction
tv_columns = df_tv.drop(columns=['Category']).columns  # Use all columns except 'Category' for prediction
best_model_tv, best_mse_tv = evaluate_model(df_tv, tv_columns, '2023-2024')
print(f"Best Model for Traditional Vehicle Sales Prediction: {best_model_tv.__class__.__name__}, MSE: {best_mse_tv}")

# 🛠 3. Forecast using the best model for Traditional Vehicle Sales
def make_forecast(best_model, df, features_columns):
    features = df[features_columns]
    forecast = best_model.predict(features)
    return forecast

# Forecast for Traditional Vehicle Sales
forecast_tv = make_forecast(best_model_tv, df_tv, tv_columns)
print(f"Forecast for Traditional Vehicle Sales: {forecast_tv}")


# In[8]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import re

# 📌 Load the cleaned Traditional Vehicle data
file_path = "C:/Users/joelr/OneDrive/Documents/Cleaned3Data/cleaned_traditional_vehicle.csv"
df_tv = pd.read_csv(file_path)

# 📌 Remove commas and convert to numeric values (Sales data)
df_tv = df_tv.replace({',': ''}, regex=True)  # Remove commas
df_tv.iloc[:, 1:] = df_tv.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')  # Convert all sales columns to numeric

# 📌 Handle missing values by filling NaN with 0
df_tv = df_tv.fillna(0)  # Or use df_tv.fillna(df_tv.mean()) for mean imputation

# 📌 Reshape the dataframe to long format for easier plotting
df_tv_long = df_tv.melt(id_vars="Category", var_name="Year", value_name="Sales")

# 📌 Convert Year to string if needed (e.g., for handling '2018-2019' format)
df_tv_long["Year"] = df_tv_long["Year"].astype(str).apply(lambda x: re.findall(r"\d{4}", x)[0] if re.findall(r"\d{4}", x) else x)

# 📌 Convert Year to numeric
df_tv_long["Year"] = df_tv_long["Year"].astype(int)

# 📌 Train a Polynomial Regression model (degree 2 for quadratic trends)
future_years = np.arange(df_tv_long["Year"].min(), df_tv_long["Year"].max() + 6).reshape(-1, 1)  # Future 5 years

predictions = []
actual_data = []

for category in df_tv_long["Category"].unique():
    df_filtered = df_tv_long[df_tv_long["Category"] == category]
    
    # 📌 Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(df_filtered["Year"].values.reshape(-1, 1))

    model = LinearRegression()
    model.fit(X_poly, df_filtered["Sales"].values)
    
    # 📌 Predict future sales
    future_X_poly = poly.transform(future_years)
    future_sales = model.predict(future_X_poly)
    
    # 📌 Store actual sales data
    for year, sales in zip(df_filtered["Year"], df_filtered["Sales"]):
        actual_data.append({"Category": category, "Year": year, "Sales": sales, "Type": "Actual"})

    # 📌 Store predicted sales data
    for year, sales in zip(future_years.flatten(), future_sales):
        predictions.append({"Category": category, "Year": year, "Sales": sales, "Type": "Predicted"})

# 📌 Convert to DataFrame
df_predictions = pd.DataFrame(predictions)
df_actual = pd.DataFrame(actual_data)

# 📌 Combine Actual and Predicted Data
df_combined = pd.concat([df_actual, df_predictions])

# 📊 Plotly Interactive Plot: Actual vs Predicted Sales
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#8e44ad', '#1abc9c', '#d35400', '#9b59b6', '#34495e']

# Create a new Plotly figure
fig = go.Figure()

# Loop through each category for both actual and predicted data
for i, category in enumerate(df_tv_long["Category"].unique()):
    df_filtered_actual = df_actual[df_actual["Category"] == category]
    df_filtered_pred = df_predictions[df_predictions["Category"] == category]

    # Plot Actual sales
    fig.add_trace(go.Scatter(
        x=df_filtered_actual["Year"], 
        y=df_filtered_actual["Sales"], 
        mode='lines+markers', 
        name=f"{category} - Actual", 
        line=dict(dash="solid", width=2, color=colors[i % len(colors)]), 
        marker=dict(size=8)
    ))

    # Plot Predicted sales
    fig.add_trace(go.Scatter(
        x=df_filtered_pred["Year"], 
        y=df_filtered_pred["Sales"], 
        mode='lines+markers', 
        name=f"{category} - Predicted", 
        line=dict(dash="dash", width=2, color=colors[(i + 1) % len(colors)]), 
        marker=dict(symbol="x", size=8)
    ))

# Customize layout
fig.update_layout(
    title="Traditional Vehicle Sales - Actual vs. Predicted",
    xaxis_title="Year",
    yaxis_title="Sales",
    legend_title="Vehicle Type",
    template="plotly_white",  # White background
    hovermode="x unified",
    margin=dict(l=40, r=40, t=40, b=40),
)

# Show the interactive chart
fig.show()


# In[9]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import re

# 📌 Load the cleaned Traditional Vehicle data
file_path = "C:/Users/joelr/OneDrive/Documents/Cleaned3Data/cleaned_traditional_vehicle.csv"
df_tv = pd.read_csv(file_path)

# 📌 Remove commas and convert to numeric values (Sales data)
df_tv = df_tv.replace({',': ''}, regex=True)  # Remove commas
df_tv.iloc[:, 1:] = df_tv.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')  # Convert all sales columns to numeric

# 📌 Handle missing values by filling NaN with 0
df_tv = df_tv.fillna(0)  # Or use df_tv.fillna(df_tv.mean()) for mean imputation

# 📌 Reshape the dataframe to long format for easier plotting
df_tv_long = df_tv.melt(id_vars="Category", var_name="Year", value_name="Sales")

# 📌 Convert Year to string if needed (e.g., for handling '2018-2019' format)
df_tv_long["Year"] = df_tv_long["Year"].astype(str).apply(lambda x: re.findall(r"\d{4}", x)[0] if re.findall(r"\d{4}", x) else x)

# 📌 Convert Year to numeric
df_tv_long["Year"] = df_tv_long["Year"].astype(int)

# 📌 Train a Polynomial Regression model (degree 2 for quadratic trends)
future_years = np.arange(df_tv_long["Year"].min(), df_tv_long["Year"].max() + 6).reshape(-1, 1)  # Future 5 years

predictions = []
actual_data = []

for category in df_tv_long["Category"].unique():
    df_filtered = df_tv_long[df_tv_long["Category"] == category]
    
    # 📌 Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(df_filtered["Year"].values.reshape(-1, 1))

    model = LinearRegression()
    model.fit(X_poly, df_filtered["Sales"].values)
    
    # 📌 Predict future sales
    future_X_poly = poly.transform(future_years)
    future_sales = model.predict(future_X_poly)
    
    # 📌 Store actual sales data
    for year, sales in zip(df_filtered["Year"], df_filtered["Sales"]):
        actual_data.append({"Category": category, "Year": year, "Sales": sales, "Type": "Actual"})

    # 📌 Store predicted sales data
    for year, sales in zip(future_years.flatten(), future_sales):
        predictions.append({"Category": category, "Year": year, "Sales": sales, "Type": "Predicted"})

# 📌 Convert to DataFrame
df_predictions = pd.DataFrame(predictions)
df_actual = pd.DataFrame(actual_data)

# 📌 Combine Actual and Predicted Data
df_combined = pd.concat([df_actual, df_predictions])

# 📊 Plotly Bar Chart: Actual vs Predicted Sales
# Use a set of bright colors
bright_colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A6', '#FFCD33', '#33FFD5', '#FF3333', '#33B5FF', '#FF5733']

# Create a new Plotly figure
fig = go.Figure()

# Loop through each category for both actual and predicted data
for i, category in enumerate(df_tv_long["Category"].unique()):
    df_filtered_actual = df_actual[df_actual["Category"] == category]
    df_filtered_pred = df_predictions[df_predictions["Category"] == category]

    # Plot Actual sales
    fig.add_trace(go.Bar(
        x=df_filtered_actual["Year"], 
        y=df_filtered_actual["Sales"], 
        name=f"{category} - Actual", 
        marker=dict(color=bright_colors[i % len(bright_colors)]), 
        opacity=0.8  # Slightly higher opacity for more vibrancy
    ))

    # Plot Predicted sales
    fig.add_trace(go.Bar(
        x=df_filtered_pred["Year"], 
        y=df_filtered_pred["Sales"], 
        name=f"{category} - Predicted", 
        marker=dict(color=bright_colors[(i + 1) % len(bright_colors)]), 
        opacity=0.8  # Slightly higher opacity for more vibrancy
    ))

# Customize layout
fig.update_layout(
    title="Traditional Vehicle Sales - Actual vs. Predicted",
    xaxis_title="Year",
    yaxis_title="Sales",
    barmode='group',  # Group bars together
    legend_title="Vehicle Type",
    template="plotly_white",  # White background
    margin=dict(l=40, r=40, t=40, b=40),
)

# Show the interactive chart
fig.show()


# In[10]:


import pandas as pd
import os
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import re

# Define file paths
parent_folder = r"C:\Users\joelr\OneDrive\Documents"
output_folder = r"C:\Users\joelr\OneDrive\Documents\CleanedData"

# Load cleaned Traditional Vehicle dataset
df_tv = pd.read_csv(os.path.join(output_folder, "cleaned_traditional_vehicle.csv"))

# Handle missing values by filling NaN with 0
df_tv = df_tv.fillna(0)

# Reshape the dataframe to long format for easier plotting
df_tv_long = df_tv.melt(id_vars="Category", var_name="Year", value_name="Sales")

# Convert Year to string if needed (e.g., for handling '2018-2019' format)
df_tv_long["Year"] = df_tv_long["Year"].astype(str).apply(lambda x: re.findall(r"\d{4}", x)[0] if re.findall(r"\d{4}", x) else x)

# Convert Year to numeric
df_tv_long["Year"] = df_tv_long["Year"].astype(int)

# Train a Polynomial Regression model (degree 2 for quadratic trends)
future_years = np.arange(df_tv_long["Year"].min(), df_tv_long["Year"].max() + 6).reshape(-1, 1)  # Future 5 years

predictions = []
actual_data = []

for category in df_tv_long["Category"].unique():
    df_filtered = df_tv_long[df_tv_long["Category"] == category]
    
    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(df_filtered["Year"].values.reshape(-1, 1))

    model = LinearRegression()
    model.fit(X_poly, df_filtered["Sales"].values)
    
    # Predict future sales
    future_X_poly = poly.transform(future_years)
    future_sales = model.predict(future_X_poly)
    
    # Store actual sales data
    for year, sales in zip(df_filtered["Year"], df_filtered["Sales"]):
        actual_data.append({"Category": category, "Year": year, "Sales": sales, "Type": "Actual"})

    # Store predicted sales data
    for year, sales in zip(future_years.flatten(), future_sales):
        predictions.append({"Category": category, "Year": year, "Sales": sales, "Type": "Predicted"})

# Convert to DataFrame
df_predictions = pd.DataFrame(predictions)
df_actual = pd.DataFrame(actual_data)

# Combine Actual and Predicted Data
df_combined = pd.concat([df_actual, df_predictions])

# Plotly Bar Chart: Actual vs Predicted Sales
bright_colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A6', '#FFCD33', '#33FFD5', '#FF3333', '#33B5FF', '#FF5733']

# Create a new Plotly figure
fig = go.Figure()

# Loop through each category for both actual and predicted data
for i, category in enumerate(df_tv_long["Category"].unique()):
    df_filtered_actual = df_actual[df_actual["Category"] == category]
    df_filtered_pred = df_predictions[df_predictions["Category"] == category]

    # Plot Actual sales
    fig.add_trace(go.Bar(
        x=df_filtered_actual["Year"], 
        y=df_filtered_actual["Sales"], 
        name=f"{category} - Actual", 
        marker=dict(color=bright_colors[i % len(bright_colors)]), 
        opacity=0.8  # Slightly higher opacity for more vibrancy
    ))

    # Plot Predicted sales
    fig.add_trace(go.Bar(
        x=df_filtered_pred["Year"], 
        y=df_filtered_pred["Sales"], 
        name=f"{category} - Predicted", 
        marker=dict(color=bright_colors[(i + 1) % len(bright_colors)]), 
        opacity=0.8  # Slightly higher opacity for more vibrancy
    ))

# Customize layout
fig.update_layout(
    title="Traditional Vehicle Sales - Actual vs. Predicted",
    xaxis_title="Year",
    yaxis_title="Sales",
    barmode='group',  # Group bars together
    legend_title="Vehicle Type",
    template="plotly_white",  # White background
    margin=dict(l=40, r=40, t=40, b=40),
)

# Show the interactive chart
fig.show()


# In[11]:


import pandas as pd
import os

# Define file paths (update if needed)
parent_folder = r"C:\Users\joelr\OneDrive\Documents"
output_folder = r"C:\Users\joelr\OneDrive\Documents\CleanedData"

# Load cleaned EV dataset
df_ev = pd.read_csv(os.path.join(output_folder, "cleaned_EV.csv"))

# 📌 Reshape the dataframe to long format for easier plotting and modeling
df_ev_long = df_ev.melt(id_vars="Category", var_name="Year", value_name="EV Sales")

# 📌 Filter out rows where 'Year' contains non-year values like 'Grand Total'
df_ev_long = df_ev_long[df_ev_long['Year'].str.contains('-')]

# 📌 Convert Year column to a numeric format (e.g., extract the first year from the year range)
df_ev_long["Year"] = df_ev_long["Year"].apply(lambda x: int(x.split('-')[0]))

# 📌 Check for missing values
df_ev_long.isna().sum()

# Handle missing values by filling NaN with 0 (or another strategy such as mean/median)
df_ev_long = df_ev_long.fillna(0)

# Now you can proceed with plotting or further analysis
# Example plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(df_ev_long["Year"], df_ev_long["EV Sales"], color="skyblue")
plt.xlabel("Year")
plt.ylabel("EV Sales")
plt.title("EV Sales by Year")
plt.show()


# In[12]:


import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# Define file paths (update if needed)
parent_folder = r"C:\Users\joelr\OneDrive\Documents"
output_folder = r"C:\Users\joelr\OneDrive\Documents\CleanedData"

# Load cleaned EV dataset
df_ev = pd.read_csv(os.path.join(output_folder, "cleaned_EV.csv"))

# 📌 Reshape the dataframe to long format for easier plotting and modeling
df_ev_long = df_ev.melt(id_vars="Category", var_name="Year", value_name="EV Sales")

# 📌 Filter out rows where 'Year' contains non-year values like 'Grand Total'
df_ev_long = df_ev_long[df_ev_long['Year'].str.contains('-')]

# 📌 Convert Year column to a numeric format (e.g., extract the first year from the year range)
df_ev_long["Year"] = df_ev_long["Year"].apply(lambda x: int(x.split('-')[0]))

# 📌 Handle missing values by filling NaN with 0
df_ev_long = df_ev_long.fillna(0)

# 📌 Polynomial Regression - Predict future sales
# Future years to predict (e.g., 2025-2028)
future_years = np.arange(df_ev_long["Year"].min(), 2029).reshape(-1, 1)

# Store actual and predicted data
predictions = []
actual_data = []

# Loop through each category to apply Polynomial Regression
for category in df_ev_long["Category"].unique():
    df_filtered = df_ev_long[df_ev_long["Category"] == category]
    
    # Polynomial Regression - Degree 2 (quadratic trend)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(df_filtered["Year"].values.reshape(-1, 1))

    model = LinearRegression()
    model.fit(X_poly, df_filtered["EV Sales"].values)
    
    # Predict future sales
    future_X_poly = poly.transform(future_years)
    future_sales = model.predict(future_X_poly)
    
    # Store actual sales data
    for year, sales in zip(df_filtered["Year"], df_filtered["EV Sales"]):
        actual_data.append({"Category": category, "Year": year, "EV Sales": sales, "Type": "Actual"})
    
    # Store predicted sales data
    for year, sales in zip(future_years.flatten(), future_sales):
        predictions.append({"Category": category, "Year": year, "EV Sales": sales, "Type": "Predicted"})

# Convert predictions and actual data to DataFrames
df_predictions = pd.DataFrame(predictions)
df_actual = pd.DataFrame(actual_data)

# Combine actual and predicted data
df_combined = pd.concat([df_actual, df_predictions])

# 📌 Plotting the Actual vs Predicted Sales for each Category
plt.figure(figsize=(12, 8))

# Loop through each category for plotting
categories = df_ev_long["Category"].unique()
for category in categories:
    df_filtered_actual = df_actual[df_actual["Category"] == category]
    df_filtered_pred = df_predictions[df_predictions["Category"] == category]
    
    # Plot actual sales
    plt.plot(df_filtered_actual["Year"], df_filtered_actual["EV Sales"], label=f"{category} - Actual", marker='o')
    
    # Plot predicted sales
    plt.plot(df_filtered_pred["Year"], df_filtered_pred["EV Sales"], label=f"{category} - Predicted", linestyle="--")

# Customize the plot
plt.xlabel("Year")
plt.ylabel("EV Sales")
plt.title("EV Sales: Actual vs Predicted by Category")
plt.legend(title="Category")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(rotation=45)  # Rotate year labels for better readability

# Show the plot
plt.tight_layout()
plt.show()


# In[13]:


import pandas as pd
import os
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# Define file paths (update if needed)
parent_folder = r"C:\Users\joelr\OneDrive\Documents"
output_folder = r"C:\Users\joelr\OneDrive\Documents\CleanedData"

# Load cleaned EV dataset
df_ev = pd.read_csv(os.path.join(output_folder, "cleaned_EV.csv"))

# 📌 Reshape the dataframe to long format for easier plotting and modeling
df_ev_long = df_ev.melt(id_vars="Category", var_name="Year", value_name="EV Sales")

# 📌 Filter out rows where 'Year' contains non-year values like 'Grand Total'
df_ev_long = df_ev_long[df_ev_long['Year'].str.contains('-')]

# 📌 Convert Year column to a numeric format (e.g., extract the first year from the year range)
df_ev_long["Year"] = df_ev_long["Year"].apply(lambda x: int(x.split('-')[0]))

# 📌 Handle missing values by filling NaN with 0
df_ev_long = df_ev_long.fillna(0)

# 📌 Polynomial Regression - Predict future sales
# Future years to predict (e.g., 2025-2028)
future_years = np.arange(df_ev_long["Year"].min(), 2029).reshape(-1, 1)

# Store actual and predicted data
predictions = []
actual_data = []

# Loop through each category to apply Polynomial Regression
category_colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A6', '#FFCD33', '#33FFD5', '#FF3333', '#33B5FF', '#FF5733']  # List of distinct colors

for i, category in enumerate(df_ev_long["Category"].unique()):
    df_filtered = df_ev_long[df_ev_long["Category"] == category]
    
    # Polynomial Regression - Degree 2 (quadratic trend)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(df_filtered["Year"].values.reshape(-1, 1))

    model = LinearRegression()
    model.fit(X_poly, df_filtered["EV Sales"].values)
    
    # Predict future sales
    future_X_poly = poly.transform(future_years)
    future_sales = model.predict(future_X_poly)
    
    # Store actual sales data
    for year, sales in zip(df_filtered["Year"], df_filtered["EV Sales"]):
        actual_data.append({"Category": category, "Year": year, "EV Sales": sales, "Type": "Actual"})
    
    # Store predicted sales data
    for year, sales in zip(future_years.flatten(), future_sales):
        predictions.append({"Category": category, "Year": year, "EV Sales": sales, "Type": "Predicted"})

# Convert predictions and actual data to DataFrames
df_predictions = pd.DataFrame(predictions)
df_actual = pd.DataFrame(actual_data)

# Combine actual and predicted data
df_combined = pd.concat([df_actual, df_predictions])

# 📌 Plotly Bar Chart: Actual vs Predicted Sales for each Category
fig = go.Figure()

# Loop through each category to add traces for actual and predicted sales
for i, category in enumerate(df_ev_long["Category"].unique()):
    df_filtered_actual = df_actual[df_actual["Category"] == category]
    df_filtered_pred = df_predictions[df_predictions["Category"] == category]

    # Plot Actual sales
    fig.add_trace(go.Bar(
        x=df_filtered_actual["Year"], 
        y=df_filtered_actual["EV Sales"], 
        name=f"{category} - Actual", 
        marker=dict(color=category_colors[i % len(category_colors)]), 
        opacity=0.8  # Slightly higher opacity for more vibrancy
    ))

    # Plot Predicted sales
    fig.add_trace(go.Bar(
        x=df_filtered_pred["Year"], 
        y=df_filtered_pred["EV Sales"], 
        name=f"{category} - Predicted", 
        marker=dict(color=category_colors[(i + 1) % len(category_colors)]), 
        opacity=0.8  # Slightly higher opacity for more vibrancy
    ))

# Customize layout
fig.update_layout(
    title="EV Sales: Actual vs Predicted by Category",
    xaxis_title="Year",
    yaxis_title="EV Sales",
    barmode='group',  # Group bars together
    legend_title="Category",
    template="plotly_white",  # White background
    margin=dict(l=40, r=40, t=40, b=40),
)

# Show the interactive chart
fig.show()


# In[14]:


import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Define file paths
parent_folder = r"C:\Users\joelr\OneDrive\Documents"
output_folder = r"C:\Users\joelr\OneDrive\Documents\CleanedData"

# Load cleaned Fuel Price dataset
df_fuel = pd.read_csv(os.path.join(output_folder, "cleaned_fuel.csv"))

# 📌 Rename columns to lowercase and strip spaces
df_fuel.columns = df_fuel.columns.str.strip().str.lower()

# 📌 Remove "annual increase" columns
df_fuel = df_fuel.drop(columns=[col for col in df_fuel.columns if "annual increase" in col])

# 📌 Extract fuel types (excluding 'year')
fuel_types = [col for col in df_fuel.columns if col != 'year']

# 📌 Convert Year column to integer
df_fuel["year"] = df_fuel["year"].astype(int)

# 📌 Train ML model for each fuel type and predict future prices
predictions = {}
models = {}
future_years = np.arange(df_fuel["year"].max() + 1, df_fuel["year"].max() + 6)

for fuel in fuel_types:
    X = df_fuel[['year']]
    y = df_fuel[fuel]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    models[fuel] = model  # Store model

    # Predict Future Prices
    future_years_df = pd.DataFrame(future_years, columns=["year"])
    future_prices = model.predict(future_years_df)
    predictions[fuel] = future_prices

# 📌 Create a DataFrame for Predictions
df_future = pd.DataFrame({"year": future_years})
for fuel in fuel_types:
    df_future[fuel] = predictions[fuel]

# 📌 Adjusted Bar Chart with **WIDER BARS**
plt.figure(figsize=(14, 7))  # Bigger chart for visibility
bar_width = 0.3  # 🔥 Increased width of bars

# Define 4 distinct colors
actual_petrol_color = "blue"
actual_diesel_color = "red"
predicted_petrol_color = "yellow"
predicted_diesel_color = "green"

# Create positions for bars
years = df_fuel["year"].tolist() + df_future["year"].tolist()
years = sorted(set(years))  # Ensure all years appear properly on x-axis
x_indexes = np.arange(len(years))

# Create mappings for actual and predicted
actual_petrol_values = [df_fuel.loc[df_fuel["year"] == year, fuel_types[0]].values[0] if year in df_fuel["year"].values else 0 for year in years]
actual_diesel_values = [df_fuel.loc[df_fuel["year"] == year, fuel_types[1]].values[0] if year in df_fuel["year"].values else 0 for year in years]

predicted_petrol_values = [df_future.loc[df_future["year"] == year, fuel_types[0]].values[0] if year in df_future["year"].values else 0 for year in years]
predicted_diesel_values = [df_future.loc[df_future["year"] == year, fuel_types[1]].values[0] if year in df_future["year"].values else 0 for year in years]

# Plot bars with **wider spacing**
plt.bar(x_indexes - bar_width * 1.5, actual_petrol_values, width=bar_width, color=actual_petrol_color, label="Actual Petrol")
plt.bar(x_indexes - bar_width * 0.5, actual_diesel_values, width=bar_width, color=actual_diesel_color, label="Actual Diesel")
plt.bar(x_indexes + bar_width * 0.5, predicted_petrol_values, width=bar_width, color=predicted_petrol_color, label="Predicted Petrol")
plt.bar(x_indexes + bar_width * 1.5, predicted_diesel_values, width=bar_width, color=predicted_diesel_color, label="Predicted Diesel")

# Labels & Title
plt.xlabel("Year")
plt.ylabel("Price (₹ per litre)")
plt.title("Fuel Price Trend (Actual & Predicted)")
plt.xticks(x_indexes, years, rotation=45)
plt.legend()
plt.show()


# In[15]:


import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Define file paths
parent_folder = r"C:\Users\joelr\OneDrive\Documents"
output_folder = r"C:\Users\joelr\OneDrive\Documents\CleanedData"

# Load cleaned Fuel Price dataset
df_fuel = pd.read_csv(os.path.join(output_folder, "cleaned_fuel.csv"))

# 📌 Rename columns to lowercase and remove spaces
df_fuel.columns = df_fuel.columns.str.strip().str.lower()

# 📌 Keep only relevant columns (Year, Petrol Price, Diesel Price)
df_fuel = df_fuel[['year', 'average petrol price (₹/litre)', 'average diesel price (₹/litre)']]

# 📌 Convert Year column to integer
df_fuel["year"] = df_fuel["year"].astype(int)

# 📌 Train ML model for each fuel type and predict future prices
future_years = np.arange(df_fuel["year"].max() + 1, df_fuel["year"].max() + 6)

predictions = {}
models = {}

for fuel in ['average petrol price (₹/litre)', 'average diesel price (₹/litre)']:
    X = df_fuel[['year']]
    y = df_fuel[fuel]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    models[fuel] = model  # Store model

    # Predict Future Prices
    future_prices = model.predict(future_years.reshape(-1, 1))
    predictions[fuel] = future_prices

# 📌 Create a DataFrame for Predictions
df_future = pd.DataFrame({"year": future_years})
for fuel in ['average petrol price (₹/litre)', 'average diesel price (₹/litre)']:
    df_future[fuel] = predictions[fuel]

# 📌 Merge Actual & Predicted Data into One CSV File
df_fuel["type"] = "Actual"
df_future["type"] = "Predicted"

df_combined = pd.concat([df_fuel, df_future])

# 📌 Save to CSV in the same folder
output_file = os.path.join(output_folder, "fuel_price_actual_predicted.csv")
df_combined.to_csv(output_file, index=False)

print(f"✅ CSV file saved successfully at: {output_file}")


# In[16]:


import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Define file paths
parent_folder = r"C:\Users\joelr\OneDrive\Documents"
output_folder = r"C:\Users\joelr\OneDrive\Documents\CleanedData"

# Load cleaned EV dataset
df_ev = pd.read_csv(os.path.join(output_folder, "cleaned_EV.csv"))

# 📌 Reshape the dataframe to long format for easier modeling
df_ev_long = df_ev.melt(id_vars="Category", var_name="Year", value_name="EV Sales")

# 📌 Filter out rows where 'Year' contains non-year values like 'Grand Total'
df_ev_long = df_ev_long[df_ev_long['Year'].str.contains('-')]

# 📌 Convert Year column to a numeric format (extract the first year from the year range)
df_ev_long["Year"] = df_ev_long["Year"].apply(lambda x: int(x.split('-')[0]))

# 📌 Handle missing values by filling NaN with 0
df_ev_long = df_ev_long.fillna(0)

# 📌 Define future years for prediction (e.g., 2025-2028)
future_years = np.arange(df_ev_long["Year"].min(), 2029).reshape(-1, 1)

# Store actual and predicted data
predictions = []
actual_data = []

# Loop through each category to apply Polynomial Regression
for category in df_ev_long["Category"].unique():
    df_filtered = df_ev_long[df_ev_long["Category"] == category]
    
    # Polynomial Regression - Degree 2 (quadratic trend)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(df_filtered["Year"].values.reshape(-1, 1))

    model = LinearRegression()
    model.fit(X_poly, df_filtered["EV Sales"].values)
    
    # Predict future sales
    future_X_poly = poly.transform(future_years)
    future_sales = model.predict(future_X_poly)
    
    # Store actual sales data
    for year, sales in zip(df_filtered["Year"], df_filtered["EV Sales"]):
        actual_data.append({"Category": category, "Year": year, "EV Sales": sales, "Type": "Actual"})
    
    # Store predicted sales data
    for year, sales in zip(future_years.flatten(), future_sales):
        predictions.append({"Category": category, "Year": year, "EV Sales": sales, "Type": "Predicted"})

# Convert predictions and actual data to DataFrames
df_predictions = pd.DataFrame(predictions)
df_actual = pd.DataFrame(actual_data)

# Combine actual and predicted data
df_combined = pd.concat([df_actual, df_predictions])

# 📌 Save to CSV in the same folder
output_file = os.path.join(output_folder, "ev_sales_actual_predicted.csv")
df_combined.to_csv(output_file, index=False)

print(f"✅ CSV file saved successfully at: {output_file}")


# In[17]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import re

# 📌 Load the cleaned Traditional Vehicle data
file_path = "C:/Users/joelr/OneDrive/Documents/CleanedData/cleaned_traditional_vehicle.csv"
df_tv = pd.read_csv(file_path)

# 📌 Remove commas and convert to numeric values (Sales data)
df_tv = df_tv.replace({',': ''}, regex=True)  # Remove commas
df_tv.iloc[:, 1:] = df_tv.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')  # Convert all sales columns to numeric

# 📌 Handle missing values by filling NaN with 0
df_tv = df_tv.fillna(0)  # Or use df_tv.fillna(df_tv.mean()) for mean imputation

# 📌 Reshape the dataframe to long format for easier plotting
df_tv_long = df_tv.melt(id_vars="Category", var_name="Year", value_name="Sales")

# 📌 Convert Year to string if needed (e.g., for handling '2018-2019' format)
df_tv_long["Year"] = df_tv_long["Year"].astype(str).apply(lambda x: re.findall(r"\d{4}", x)[0] if re.findall(r"\d{4}", x) else x)

# 📌 Convert Year to numeric
df_tv_long["Year"] = df_tv_long["Year"].astype(int)

# 📌 Train a Polynomial Regression model (degree 2 for quadratic trends)
future_years = np.arange(df_tv_long["Year"].min(), df_tv_long["Year"].max() + 6).reshape(-1, 1)  # Future 5 years

predictions = []
actual_data = []

for category in df_tv_long["Category"].unique():
    df_filtered = df_tv_long[df_tv_long["Category"] == category]
    
    # 📌 Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(df_filtered["Year"].values.reshape(-1, 1))

    model = LinearRegression()
    model.fit(X_poly, df_filtered["Sales"].values)
    
    # 📌 Predict future sales
    future_X_poly = poly.transform(future_years)
    future_sales = model.predict(future_X_poly)
    
    # 📌 Store actual sales data
    for year, sales in zip(df_filtered["Year"], df_filtered["Sales"]):
        actual_data.append({"Category": category, "Year": year, "Sales": sales, "Type": "Actual"})

    # 📌 Store predicted sales data
    for year, sales in zip(future_years.flatten(), future_sales):
        predictions.append({"Category": category, "Year": year, "Sales": sales, "Type": "Predicted"})

# 📌 Convert to DataFrame
df_predictions = pd.DataFrame(predictions)
df_actual = pd.DataFrame(actual_data)

# 📌 Combine Actual and Predicted Data
df_combined = pd.concat([df_actual, df_predictions])

# 📌 Export the combined actual and predicted sales data to CSV
export_path = "C:/Users/joelr/OneDrive/Documents/CleanedData/traditional_vehicle_sales_actual_predicted.csv"
df_combined.to_csv(export_path, index=False)

print(f"✅ Data successfully exported to: {export_path}")

# 📊 Plotly Bar Chart: Actual vs Predicted Sales
# Use a set of bright colors
bright_colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A6', '#FFCD33', '#33FFD5', '#FF3333', '#33B5FF', '#FF5733']

# Create a new Plotly figure
fig = go.Figure()

# Loop through each category for both actual and predicted data
for i, category in enumerate(df_tv_long["Category"].unique()):
    df_filtered_actual = df_actual[df_actual["Category"] == category]
    df_filtered_pred = df_predictions[df_predictions["Category"] == category]

    # Plot Actual sales
    fig.add_trace(go.Bar(
        x=df_filtered_actual["Year"], 
        y=df_filtered_actual["Sales"], 
        name=f"{category} - Actual", 
        marker=dict(color=bright_colors[i % len(bright_colors)]), 
        opacity=0.8  # Slightly higher opacity for more vibrancy
    ))

    # Plot Predicted sales
    fig.add_trace(go.Bar(
        x=df_filtered_pred["Year"], 
        y=df_filtered_pred["Sales"], 
        name=f"{category} - Predicted", 
        marker=dict(color=bright_colors[(i + 1) % len(bright_colors)]), 
        opacity=0.8  # Slightly higher opacity for more vibrancy
    ))

# Customize layout
fig.update_layout(
    title="Traditional Vehicle Sales - Actual vs. Predicted",
    xaxis_title="Year",
    yaxis_title="Sales",
    barmode='group',  # Group bars together
    legend_title="Vehicle Type",
    template="plotly_white",  # White background
    margin=dict(l=40, r=40, t=40, b=40),
)

# Show the interactive chart
fig.show()


# In[ ]:





# In[ ]:




