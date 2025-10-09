import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ===== LOAD DATASET =====
CSV_FILENAME = 'Student_Performance.csv'

try:
    df = pd.read_csv(CSV_FILENAME)
    print("✅ Dataset loaded successfully!")
    
except FileNotFoundError:
    print(f"❌ File '{CSV_FILENAME}' not found!")
    print("\nPlease:")
    print("1. Download the CSV from Kaggle")
    print("2. Place it in the same folder as this script")
    print("3. Update CSV_FILENAME variable if needed")
    exit()

# ===== EXPLORE DATASET =====
print(f"\n{'='*60}")
print("DATASET OVERVIEW")
print(f"{'='*60}")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn names:")
print(df.columns.tolist())
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nBasic statistics:")
print(df.describe())

# ===== CONFIGURE YOUR VARIABLES =====
X_COLUMN = 'Hours Studied'      # Independent variable (predictor)
Y_COLUMN = 'Performance Index'   # Dependent variable (target)


# ===== VALIDATE COLUMNS =====
if X_COLUMN not in df.columns:
    print(f"\n❌ ERROR: Column '{X_COLUMN}' not found!")
    print(f"Available columns: {df.columns.tolist()}")
    print("\nPlease update X_COLUMN variable in the code")
    exit()

if Y_COLUMN not in df.columns:
    print(f"\n❌ ERROR: Column '{Y_COLUMN}' not found!")
    print(f"Available columns: {df.columns.tolist()}")
    print("\nPlease update Y_COLUMN variable in the code")
    exit()

# ===== DATA PREPROCESSING =====
# Remove missing values
df_clean = df[[X_COLUMN, Y_COLUMN]].dropna()
print(f"\n🧹 Data cleaning: {len(df_clean)} rows remaining from {len(df)}")

# Prepare data for regression
X = df_clean[[X_COLUMN]].values  # 2D array
y = df_clean[Y_COLUMN].values    # 1D array

print(f"\n📊 Final dataset:")
print(f"Number of observations: {len(X)}")
print(f"X ({X_COLUMN}) range: {X.min():.2f} to {X.max():.2f}")
print(f"Y ({Y_COLUMN}) range: {y.min():.2f} to {y.max():.2f}")

# ===== LINEAR REGRESSION MODEL =====
model = LinearRegression()
model.fit(X, y)

# Get coefficients
coefficient = model.coef_[0]
intercept = model.intercept_

# Make predictions
y_pred = model.predict(X)

# Calculate metrics
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

# ===== PRINT RESULTS =====
print(f"\n{'='*60}")
print("LINEAR REGRESSION RESULTS")
print(f"{'='*60}")
print(f"Equation: {Y_COLUMN} = {coefficient:.4f} × {X_COLUMN} + {intercept:.4f}")
print(f"\nCoefficient (slope): {coefficient:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"\nR² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Interpretation
print(f"\n{'='*60}")
print("INTERPRETATION")
print(f"{'='*60}")
if coefficient > 0:
    print(f"✓ Positive relationship: As {X_COLUMN} increases by 1 unit,")
    print(f"  {Y_COLUMN} increases by {coefficient:.4f} units on average.")
else:
    print(f"✓ Negative relationship: As {X_COLUMN} increases by 1 unit,")
    print(f"  {Y_COLUMN} decreases by {abs(coefficient):.4f} units on average.")

if r2 >= 0.8:
    print(f"✓ R² = {r2:.4f}: Excellent model fit! Strong predictive power.")
elif r2 >= 0.6:
    print(f"✓ R² = {r2:.4f}: Good model fit. Reasonable predictions.")
elif r2 >= 0.4:
    print(f"⚠ R² = {r2:.4f}: Moderate fit. Limited predictive ability.")
else:
    print(f"⚠ R² = {r2:.4f}: Weak fit. Poor predictive ability.")

# ===== VISUALIZATION (SINGLE PLOT) =====
plt.figure(figsize=(12, 8))

# Scatter plot - actual data
plt.scatter(X, y, alpha=0.6, s=60, color='steelblue', 
           edgecolor='white', linewidth=0.8,
           label='Actual Data')

# Regression line
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='red', linewidth=3,
         label=f'Linear Regression\n{Y_COLUMN} = {coefficient:.4f}×{X_COLUMN} + {intercept:.4f}')

# Labels and title
plt.xlabel(X_COLUMN, fontsize=14, fontweight='bold')
plt.ylabel(Y_COLUMN, fontsize=14, fontweight='bold')
plt.title(f'Linear Regression: {Y_COLUMN} vs {X_COLUMN}\nR² = {r2:.4f}', 
         fontsize=16, fontweight='bold', pad=20)

# Legend
plt.legend(fontsize=11, loc='best', framealpha=0.9)

# Grid
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# Add text box with metrics
textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nSamples: {len(X)}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', bbox=props)

# Styling
plt.tight_layout()
plt.show()

print(f"\n✅ Analysis complete!")

# ===== OPTIONAL: MAKE PREDICTIONS =====
print(f"\n{'='*60}")
print("EXAMPLE PREDICTIONS")
print(f"{'='*60}")

# Generate sample predictions
x_min, x_max = X.min()[0], X.max()[0]
sample_values = np.linspace(x_min, x_max, 5)

for x_val in sample_values:
    y_predicted = model.predict([[x_val]])[0]
    print(f"When {X_COLUMN} = {x_val:.2f} → Predicted {Y_COLUMN} = {y_predicted:.2f}")

# ===== SAVE RESULTS =====
results_df = df_clean.copy()
results_df['Predicted'] = y_pred
results_df['Residual'] = y - y_pred

results_df.to_csv('regression_results.csv', index=False)
print(f"\n💾 Results saved to 'regression_results.csv'")