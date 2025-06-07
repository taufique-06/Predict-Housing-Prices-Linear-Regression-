import pandas as pd # This will handle table like data
import numpy as np # Math Stufd
import matplotlib.pyplot as plt # Draw charts and plots
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

california = fetch_california_housing();
df = pd.DataFrame(california.data, columns=california.feature_names); # covert it in a table
df['PRICE'] = california.target  # Target column is 'PRICE' which is the value we want to predict

df.head(); # Shows first 5 rows

df.describe(); 

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm') #df.corr() strongly looks at every column in our dataset and checks how strongly each one is related to each other
plt.title("Correlation Heatmap")
plt.show()

X = df.drop("PRICE", axis=1);
y = df["PRICE"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42);

model = RandomForestRegressor(random_state=42, n_estimators=500)
model.fit(X_train, y_train);


y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()



