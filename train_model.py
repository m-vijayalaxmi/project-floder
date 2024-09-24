import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import seaborn as sns

# Load and clean data
data = pd.read_csv("data.csv")
data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
data['Date'] = pd.to_datetime(data['Date'], format='%b %d, %Y')
data['Volume'] = data['Volume'].str.replace(',', '').astype(float)
price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
data.head(10)
data.shape
data.info()
data.describe()
data[price_columns] = data[price_columns].apply(pd.to_numeric, errors='coerce')

from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
for i in price_columns:
    data[i]=mm.fit_transform(data[[i]])
data.head()
# Drop rows with missing values
data = data.dropna()
# sns.pairplot(data)
sns.histplot(data=data,x='Close')
sns.lineplot(data=data,x=data.index,y='Close')

# Select features 
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'stock_price_model.pkl')
print("Model trained and saved as 'stock_price_model.pkl'")

# Make predictions on the test set
y_pred = model.predict(X_test)
import matplotlib.pyplot as plt

# Plotting predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel('Actual Stock Prices')
plt.ylabel('Predicted Stock Prices')
plt.title('Actual vs Predicted Stock Prices')
plt.grid(True)
plt.show()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")
