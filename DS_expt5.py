import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# #1: Load dataset
df = pd.read_csv("cdsp-rainfall.csv")
# print("Dataset Shape:", df.shape)
# print(df.head())

# Step 2 & 3: Perform basic data cleaning first , Identify independent and dependent variables
df = df.drop('id', axis=1, errors='ignore') # Drop useless identifier

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df = df.drop('date', axis=1)

categorical_cols = ['state_name', 'district_name', 'station_name']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

df.fillna(df.median(), inplace=True)

target = 'average_rainfall'
y = df[target] 
X = df.drop(target, axis=1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Features (X):", X.columns.tolist())
print("Target (y):", target)

# Step 4: Correlation analysis
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Step 5: Filter Method (SelectKBest using f_regression)
num_features = 4
filter_selector = SelectKBest(score_func=f_regression, k=num_features)
filter_selector.fit(X_train, y_train)

filter_features = X.columns[filter_selector.get_support()].tolist()
print(f"Top {num_features} features (Filter Method): {filter_features}")

# Step 6: Wrapper Method (Recursive Feature Elimination)
rfe_estimator = LinearRegression()
rfe_selector = RFE(estimator=rfe_estimator, n_features_to_select=num_features, step=1)
rfe_selector.fit(X_train, y_train)

rfe_features = X.columns[rfe_selector.support_].tolist()
print(f"Top {num_features} features (RFE): {rfe_features}")

# Step 7: Embedded Method (Random Forest)
rf_selector = RandomForestRegressor(n_estimators=20, random_state=42, max_depth=5)
rf_selector.fit(X_train, y_train)

importances = pd.Series(rf_selector.feature_importances_, index=X.columns)
embedded_features = importances.nlargest(num_features).index.tolist()

plt.figure(figsize=(8, 5))
importances.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title("Random Forest Feature Importances")
plt.ylabel("Importance Score")
plt.show()
print(f"Top {num_features} features (Embedded Method): {embedded_features}")

# Step 8: Compare selected features
print("Comparison of Selected Features:")
print(f"1. Filter Method  : {filter_features}")
print(f"2. Wrapper (RFE)  : {rfe_features}")
print(f"3. Embedded (Tree): {embedded_features}")

final_selected_features = embedded_features

# Step 9: Train and evaluate model using ALL features
rf_all = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=7)
rf_all.fit(X_train, y_train)

y_pred_all = rf_all.predict(X_test)
mse_all = mean_squared_error(y_test, y_pred_all)
r2_all = r2_score(y_test, y_pred_all)

print("--- Model with ALL Features ---")
print(f"MSE: {mse_all:.2f}")
print(f"R-squared: {r2_all:.4f}")

# Step 10: Train and evaluate model using SELECTED features
X_train_sel = X_train[final_selected_features]
X_test_sel = X_test[final_selected_features]

rf_selected = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=7)
rf_selected.fit(X_train_sel, y_train)

y_pred_sel = rf_selected.predict(X_test_sel)
mse_sel = mean_squared_error(y_test, y_pred_sel)
r2_sel = r2_score(y_test, y_pred_sel)

print("--- Model with SELECTED Features ---")
print(f"MSE: {mse_sel:.2f}")
print(f"R-squared: {r2_sel:.4f}")