import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

CSV_FILENAME = 'Student_Performance.csv'
df = pd.read_csv(CSV_FILENAME)

X = df.drop(columns=['Performance Index', 'Extracurricular Activities'])
y = df['Performance Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# Get coefficients
coefficient = model.coef_[0]
intercept = model.intercept_

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)


# Visualization
plt.figure(figsize=(14, 8))

# Scatter plot - actual vs predicted
plt.scatter(y_test, y_pred, alpha=0.6, s=80, color='steelblue', 
           edgecolor='white', linewidth=0.8,
           label='Predictions')

# Perfect prediction line (y = y_pred)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=3, 
         label='Perfect Prediction')

# Labels and title
plt.xlabel('Actual Performance Index', fontsize=14, fontweight='bold')
plt.ylabel('Predicted Performance Index', fontsize=14, fontweight='bold')
plt.title('Multiple Linear Regression: Actual vs Predicted Performance\nStudent Performance Analysis', 
         fontsize=16, fontweight='bold', pad=20)

# Legend
plt.legend(fontsize=12, loc='upper left', framealpha=0.95)

# Grid
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# Add text box with metrics
textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMSE = {mse:.4f}\nSamples = {len(y_test)}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
plt.text(0.98, 0.05, textstr, transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='bottom', horizontalalignment='right',
         bbox=props, family='monospace')

# Styling
plt.tight_layout()
plt.show()


print(f"\n{'='*30}")
print("LINEAR REGRESSION RESULTS")
print(f"\nCoefficient (slope): {coefficient:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"\nR² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")


# Data for JSON
results = {
    'r2': float(r2),
    'rmse': float(rmse),
    'mse': float(mse),
    'n_samples': int(len(y_test)),
    'actual': y_test.values.tolist(),
    'predicted': y_pred.tolist(),
    'features': X.columns.tolist()
}

# ===== SAVE TO JSON =====
with open('regression_data.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nData saved to 'regression_data.json'")