#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[3]:


data = pd.read_csv("revenue_lift_9pct_30000.csv")
data


# In[5]:


data.shape


# In[7]:


data.isnull().sum()


# In[9]:


data.duplicated().sum()


# In[11]:


data.dtypes


# In[13]:


data.describe()


# In[15]:


# Negative or invalid values
print("\nInvalid values check:")
print("Units Sold <= 0:", (data["Units Sold"] <= 0).sum())
print("Price <= 0:", (data["Price"] <= 0).sum())
print("Revenue <= 0:", (data["Revenue"] <= 0).sum())
print("Cost Price <= 0:", (data["Cost Price"] <= 0).sum())
print("Stock Level < 0:", (data["Stock Level"] < 0).sum())


# In[17]:


#revenue lift
data["Baseline_Revenue"] = (data["Cost Price"] * 1.20) * data["Units Sold"]

revenue_lift = (
    (data["Revenue"].sum() - data["Baseline_Revenue"].sum())
    / data["Baseline_Revenue"].sum()
) * 100

print(f"Revenue Lift: {revenue_lift:.2f}%")


# In[19]:


#Profit Margin
data["Profit"] = data["Revenue"] - (data["Cost Price"] * data["Units Sold"])

profit_margin = (data["Profit"].sum() / data["Revenue"].sum()) * 100

print(f"Profit Margin: {profit_margin:.2f}%")


# In[21]:


#Conversion Rate
conversion_rate = (data["Units Sold"].sum() / data["Visitors"].sum()) * 100

print(f"Conversion Rate: {conversion_rate:.2f}%")


# In[23]:


#Inventory Turnover
inventory_turnover = data["Units Sold"].sum() / data["Stock Level"].mean()

print(f"Inventory Turnover: {inventory_turnover:.2f}")


# In[25]:


data["Date"] = pd.to_datetime(data["Date"])

data["Day"] = data["Date"].dt.day
data["Month"] = data["Date"].dt.month
data["Year"] = data["Date"].dt.year
data["DayOfWeek"] = data["Date"].dt.weekday
data["Weekend"] = data["DayOfWeek"].isin([5, 6]).astype(int)

# Season (India)
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Rainy"
    else:
        return "Autumn"

data["Season"] = data["Month"].apply(get_season)

# Holiday flag (simple proxy)
data["Holiday"] = data["Weekend"]


# In[27]:


data["Lag_Sales_1"] = data.groupby("Product ID")["Units Sold"].shift(1)
data["Lag_Sales_7"] = data.groupby("Product ID")["Units Sold"].shift(7)
data["Lag_Sales_30"] = data.groupby("Product ID")["Units Sold"].shift(30)

data["Rolling_7"] = data.groupby("Product ID")["Units Sold"].rolling(7).mean().reset_index(0, drop=True)
data["Rolling_30"] = data.groupby("Product ID")["Units Sold"].rolling(30).mean().reset_index(0, drop=True)

data["Demand_Volatility"] = data.groupby("Product ID")["Units Sold"].rolling(7).std().reset_index(0, drop=True)


# In[28]:


# ---------- PRICE CHANGE ----------
data = data.sort_values(["Product ID", "Date"])

data["Lag_Price_1"] = data.groupby("Product ID")["Price"].shift(1)

data["Price_Change_%"] = (
    (data["Price"] - data["Lag_Price_1"]) / data["Lag_Price_1"]
) * 100


# In[29]:


data["Elasticity"] = data["Price_Change_%"] / (
    (data["Units Sold"] - data["Lag_Sales_1"]) / data["Lag_Sales_1"]
)

def elasticity_class(x):
    if abs(x) > 1:
        return "High"
    elif abs(x) > 0.5:
        return "Medium"
    else:
        return "Low"

data["Elasticity_Class"] = data["Elasticity"].apply(elasticity_class)


# In[33]:


data["Competitor_Diff"] = data["Price"] - data["Competitor Price"]
data["Competitor_Index"] = data["Price"] / data["Competitor Price"]
data["Competitor_Cheaper"] = (data["Competitor Price"] < data["Price"]).astype(int)


# In[35]:


data["Inventory_Ratio"] = data["Units Sold"] / data["Stock Level"]
data["Days_To_Stockout"] = data["Stock Level"] / data["Units Sold"]

data["Low_Stock"] = (data["Stock Level"] < data["Stock Level"].quantile(0.25)).astype(int)
data["Over_Stock"] = (data["Stock Level"] > data["Stock Level"].quantile(0.75)).astype(int)


# In[37]:


data["Profit_Per_Unit"] = data["Profit"] / data["Units Sold"]
data["Profit_Margin_Feature"] = (data["Profit"] / data["Revenue"]) * 100


# In[39]:


data["Discount_%"] = (
    (data["Competitor Price"] - data["Price"]) / data["Competitor Price"]
) * 100


# In[41]:


data["Weekend_Price"] = data["Weekend"] * data["Price"]

data["Season_Discount"] = data["Discount_%"] * (
    data["Season"] == "Summer"
).astype(int)

data["Inventory_Price"] = data["Stock Level"] * data["Price"]


# In[43]:


data.columns


# In[45]:


data.isnull().sum()


# In[47]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data["Product_ID_Enc"] = le.fit_transform(data["Product ID"])
data["Category_Enc"] = le.fit_transform(data["Category"])
data["Pricing_Type_Enc"] = le.fit_transform(data["Pricing_Type"])


# In[49]:


data.drop_duplicates(inplace=True)
data.fillna(0, inplace=True)


# In[51]:


data.replace([float("inf"), -float("inf")], 0, inplace=True)


# In[53]:


numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns


# In[55]:


Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

data = data[~((data[numeric_cols] < (Q1 - 1.5 * IQR)) |
              (data[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[57]:


print("Rows after outlier removal:", data.shape[0])


# In[59]:


#milestone 3


# In[149]:


import matplotlib.pyplot as plt

plt.scatter(data["Price"], data["Units Sold"])
plt.xlabel("Price")
plt.ylabel("Units Sold")
plt.title("Price vs Units Sold")
plt.show()


# In[70]:


# Price Distribution
import seaborn as sns
plt.figure(figsize=(6,4))
sns.histplot(data['Price'], bins=30, kde=True)
plt.title("Price Distribution")
plt.show()


# In[74]:


# Units Sold Distribution
plt.figure(figsize=(6,4))
sns.histplot(data['Units Sold'], bins=30, kde=True)
plt.title("Units Sold Distribution")
plt.show()


# In[78]:


data['Date'] = pd.to_datetime(data['Date'])
daily_revenue = data.groupby(data['Date'].dt.date)['Revenue'].sum()

daily_revenue.plot(figsize=(10,4))
plt.title("Daily Revenue Trend")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.show()


# In[80]:


data.set_index("Date").resample("M")["Revenue"].sum().plot(figsize=(10,4))
plt.title("Monthly Revenue Trend")
plt.show()


# In[82]:


cols = ["Price", "Units Sold", "Revenue", "Profit",
        "Stock Level", "Competitor Price", "Inventory Turnover"]

sns.heatmap(data[cols].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# In[84]:


data[cols].boxplot(figsize=(12,5))
plt.show()


# ### (Milestone 4)PriceOptima – Baseline Rule-Based Pricing Engine

# In[61]:


#Weekend demand is higher → price increase
#Festival/peak season → price increase
#Low demand period → discount
#Low stock → price increase
#High stock → discount


# In[63]:


import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format


# In[65]:


data["rule_price"] = data["Price"]


# In[67]:


# Weekend demand surge
data.loc[data["Weekend"] == 1, "rule_price"] *= 1.10

# Festival / peak season pricing
data.loc[data["Holiday"] == 1, "rule_price"] *= 1.15

# Month-end increase
data.loc[data["Date"].dt.day >= 25, "rule_price"] *= 1.05


# In[69]:


low_stock_threshold = data["Stock Level"].quantile(0.25)
high_stock_threshold = data["Stock Level"].quantile(0.75)

# Low stock → increase price
data.loc[data["Stock Level"] < low_stock_threshold, "rule_price"] *= 1.15

# High stock → small discount only
data.loc[data["Stock Level"] > high_stock_threshold, "rule_price"] *= 0.97


# In[71]:


data["Static_Revenue"] = data["Price"] * data["Units Sold"]
data["Rule_Based_Revenue"] = data["rule_price"] * data["Units Sold"]

revenue_comparison = data[["Static_Revenue", "Rule_Based_Revenue"]].sum()

revenue_comparison_usd = revenue_comparison.apply(lambda x: f"${x:,.2f}")
revenue_comparison_usd

print(revenue_comparison)


# In[73]:


revenue_lift = (
    (revenue_comparison["Rule_Based_Revenue"] -
     revenue_comparison["Static_Revenue"])
    / revenue_comparison["Static_Revenue"]
) * 100

print(f"Revenue Lift: {revenue_lift:.2f}%")


# In[75]:


data[["Price", "rule_price", "Units Sold"]].head()


# ### Milestone 5 – Advanced Model Development

# In[78]:


# Drop rows with missing values
data = data.dropna().reset_index(drop=True)


# In[80]:


#Encode Categorical Variables
from sklearn.preprocessing import LabelEncoder

cat_cols = ["Product ID", "Product Name", "Category", "Pricing_Type"]

le = LabelEncoder()
for col in cat_cols:
    data[col] = le.fit_transform(data[col])


# In[82]:


#Define Features (X) and Target (y)
features = [
    "Price", "Competitor Price", "Stock Level",
    "Weekend", "Holiday",
    "Lag_Sales_1", "Lag_Sales_7",
    "Rolling_7", "Rolling_30",
    "Inventory_Ratio", "Profit Margin %"
]

X = data[features]
y = data["Units Sold"]


# In[84]:


#Time-Based Train-Test Split
split_date = data["Date"].quantile(0.8)

X_train = X[data["Date"] <= split_date]
X_test  = X[data["Date"] > split_date]

y_train = y[data["Date"] <= split_date]
y_test  = y[data["Date"] > split_date]


# In[86]:


#XGBoost Regressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

xgb = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)

print("XGBoost MAE:", xgb_mae)


# In[88]:


#LightGBM Regressor
from lightgbm import LGBMRegressor

lgb = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

lgb.fit(X_train, y_train)

lgb_pred = lgb.predict(X_test)
lgb_mae = mean_absolute_error(y_test, lgb_pred)

print("LightGBM MAE:", lgb_mae)


# In[89]:


#ML-Based Dynamic Pricing
data_test = data[data["Date"] > split_date].copy()

data_test["Predicted_Demand"] = lgb_pred

data_test["ml_price"] = data_test["Price"]

# High demand → increase price
data_test.loc[
    data_test["Predicted_Demand"] > data_test["Units Sold"].mean(),
    "ml_price"
] *= 1.05

# Low demand → decrease price
data_test.loc[
    data_test["Predicted_Demand"] <= data_test["Units Sold"].mean(),
    "ml_price"
] *= 0.95


# In[92]:


#Revenue Calculation
# Static pricing revenue
data_test["Static_Revenue"] = data_test["Price"] * data_test["Units Sold"]

# Rule-based revenue (from Milestone 4)
data_test["Rule_Based_Revenue"] = data_test["rule_price"] * data_test["Units Sold"]

# ML-based revenue
data_test["ML_Revenue"] = data_test["ml_price"] * data_test["Units Sold"]


# In[94]:


#Revenue Comparison Table
revenue_summary = data_test[
    ["Static_Revenue", "Rule_Based_Revenue", "ML_Revenue"]
].sum()


# In[96]:


revenue_summary


# In[98]:


#Revenue Lift Calculation
ml_revenue_lift = (
    (revenue_summary["ML_Revenue"] - revenue_summary["Static_Revenue"])
    / revenue_summary["Static_Revenue"]
) * 100

print(f"ML Revenue Lift: {ml_revenue_lift:.2f}%")


# In[104]:


import joblib

joblib.dump(lgb, "priceoptima_lgb_model.joblib")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




