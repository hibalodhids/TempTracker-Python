import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Simulated dataset (or load from CSV)
data = {
    'Area': [1000, 1500, 1800, 2400, 3000, 3500, 4000],
    'Bedrooms': [2, 3, 3, 4, 4, 5, 5],
    'Age': [10, 15, 20, 5, 8, 12, 18],
    'Price': [200000, 250000, 270000, 350000, 400000, 450000, 500000]
}

df = pd.DataFrame(data)
print(df)


# step 2 data pre-processing

# Features and Target
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

#  Step 4: Split into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Step 5: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 🔮 Step 6: Make Predictions
y_pred = model.predict(X_test)

# 🧾 Step 7: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Predict a New House Price
new_house = pd.DataFrame([[1500, 4, 15]], columns=["Area", "Bedrooms", "Age"])
predicted_price = model.predict(new_house)[0]

print(f"\n💡 Predicted Price for 1500 sqft, 4-bed, 15-year old house: {predicted_price:,.2f}")


plt.scatter(df['Area'], df['Price'], color='blue', label='Actual Price')
plt.plot(df['Area'], model.predict(df[['Area', 'Bedrooms', 'Age']]), color='red', label='Regression Line')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('House Price vs Area')
plt.legend()
plt.show()