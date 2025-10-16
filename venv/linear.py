import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ===== LOAD DATASET =====
CSV_FILENAME = 'Student_Performance.csv'

df = pd.read_csv(CSV_FILENAME)
df_clean = df.dropna()

# Split features and target
X = df_clean.drop(columns=['Performance Index', 'Extracurricular Activities'])
y = df_clean['Performance Index']

print(f"\nFeatures used: {X.columns.tolist()}")
print(f"Target variable: Performance Index")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")

# ===== LINEAR REGRESSION MODEL =====
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# ===== PRINT RESULTS =====
print(f"\n{'='*60}")
print("LINEAR REGRESSION RESULTS")
print(f"{'='*60}")
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

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

# ===== CREATE COEFFICIENTS TABLE =====
coefficients_data = []
coefficients_data.append({
    'feature': 'Intercept',
    'coefficient': float(model.intercept_),
    'interpretation': 'Base performance value'
})

for i, feature in enumerate(X.columns):
    coefficients_data.append({
        'feature': feature,
        'coefficient': float(model.coef_[i]),
        'interpretation': f'Effect of {feature} on performance'
    })

# ===== PREPARE DATASET TABLE (first 10 rows) =====
dataset_preview = []
for idx in range(min(10, len(df))):
    row_data = {}
    for col in df.columns:
        value = df.iloc[idx][col]
        # Check if value is NaN
        if pd.isna(value):
            row_data[col] = None
        # Try to convert to float if it's numeric
        elif isinstance(value, (int, float, np.integer, np.floating)):
            row_data[col] = float(value)
        # Keep as string if it's text
        else:
            row_data[col] = str(value)
    dataset_preview.append(row_data)

# ===== PREPARE DATA FOR JSON =====
results = {
    'r2': float(r2),
    'rmse': float(rmse),
    'mse': float(mse),
    'n_samples': int(len(y_test)),
    'actual': y_test.values.tolist(),
    'predicted': y_pred.tolist(),
    'coefficients': coefficients_data,
    'features': X.columns.tolist(),
    'dataset_preview': dataset_preview,
    'total_rows': len(df)
}

# ===== GENERATE HTML FILE WITH EMBEDDED DATA =====
html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multiple Linear Regression</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }

        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5em;
        }

        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 1.1em;
        }

        .chart-container {
            position: relative;
            height: 500px;
            margin-bottom: 40px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 15px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }

        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
        }

        .metric-card h3 {
            font-size: 0.9em;
            margin-bottom: 10px;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .metric-card .value {
            font-size: 2.5em;
            font-weight: bold;
        }

        .coefficients-section {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            border-left: 5px solid #667eea;
        }

        .coefficients-section h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.5em;
        }

        .coefficients-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
        }

        .coefficients-table th,
        .coefficients-table td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        .coefficients-table th {
            background: #667eea;
            color: white;
            font-weight: bold;
        }

        .coefficients-table tr:hover {
            background: #e8eaf6;
        }

        .coefficients-table tr:last-child td {
            border-bottom: none;
        }

        .explanation {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            border-left: 5px solid #667eea;
        }

        .explanation h2 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.3em;
            margin-top: 20px;
        }

        .explanation h2:first-child {
            margin-top: 0;
        }

        .explanation p {
            color: #555;
            line-height: 1.8;
            margin-bottom: 15px;
        }

        .highlight {
            background: #fff3cd;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
        }

        .footer {
            text-align: center;
            margin-top: 30px;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Multiple Linear Regression Analysis</h1>
        <div class="subtitle">Student Performance Prediction Model</div>

        <div class="chart-container">
            <canvas id="regressionChart"></canvas>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <h3>R² Score</h3>
                <div class="value" id="r2score">-</div>
                <small>Model Accuracy</small>
            </div>
            <div class="metric-card">
                <h3>RMSE</h3>
                <div class="value" id="rmse">-</div>
                <small>Average Error</small>
            </div>
            <div class="metric-card">
                <h3>MSE</h3>
                <div class="value" id="mse">-</div>
                <small>Mean Squared Error</small>
            </div>
            <div class="metric-card">
                <h3>Samples</h3>
                <div class="value" id="samples">-</div>
                <small>Test Data Points</small>
            </div>
        </div>

        <div class="coefficients-section">
            <h2>Model Coefficients</h2>
            <table class="coefficients-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Coefficient</th>
                        <th>Interpretation</th>
                    </tr>
                </thead>
                <tbody id="coefficientsTable">
                    <!-- Generated by JavaScript -->
                </tbody>
            </table>
        </div>

        <div class="coefficients-section">
            <h2>Dataset Preview (First 10 Rows)</h2>
            <p style="color: #666; margin-bottom: 15px; font-size: 0.95em;">
                Showing <span id="previewCount">-</span> rows out of <span id="totalRows">-</span> total rows
            </p>
            <div style="overflow-x: auto;">
                <table class="coefficients-table" id="datasetTable">
                    <thead id="datasetTableHead">
                        <!-- Generated by JavaScript -->
                    </thead>
                    <tbody id="datasetTableBody">
                        <!-- Generated by JavaScript -->
                    </tbody>
                </table>
            </div>
        </div>

        <div class="explanation">
            <h2>Understanding the Graph</h2>
            <p>
                The <span class="highlight">Actual vs Predicted</span> scatter plot shows how well our model performs. 
                Each blue dot represents a student, with X-axis showing <span class="highlight">actual performance</span> 
                and Y-axis showing <span class="highlight">predicted performance</span>.
            </p>

            <h2>Perfect Prediction Line</h2>
            <p>
                The red line represents <span class="highlight">y = x</span> (perfect predictions). 
                When actual equals predicted, points fall on this line. 
                Points <span class="highlight">close to the line</span> = accurate predictions ✓
                <br>
                Points <span class="highlight">far from the line</span> = prediction errors ✗
            </p>

            <h2>Metrics Explained</h2>
            <p>
                <strong>R² Score:</strong> Proportion of variance explained (0 to 1). Higher is better.<br>
                <strong>RMSE:</strong> Average prediction error in original units. Lower is better.<br>
                <strong>MSE:</strong> Mean of squared errors. RMSE = √MSE.
            </p>

            <h2>Model Quality</h2>
            <p>
                R² ≥ 0.8 = Excellent | R² 0.6-0.8 = Good | R² 0.4-0.6 = Moderate | R² < 0.4 = Weak
            </p>
        </div>

        <div class="footer">
            <p>Multiple Linear Regression | Student Performance Analysis</p>
        </div>
    </div>

    <script>
        // Data embedded directly in HTML
        const data = DATA_PLACEHOLDER;

        // Display results
        document.getElementById('r2score').textContent = data.r2.toFixed(4);
        document.getElementById('rmse').textContent = data.rmse.toFixed(4);
        document.getElementById('mse').textContent = data.mse.toFixed(4);
        document.getElementById('samples').textContent = data.n_samples;

        // Fill coefficients table
        const tableBody = document.getElementById('coefficientsTable');
        data.coefficients.forEach(coef => {
            const row = tableBody.insertRow();
            row.innerHTML = `
                <td><strong>${coef.feature}</strong></td>
                <td>${coef.coefficient.toFixed(6)}</td>
                <td>${coef.interpretation}</td>
            `;
        });

        // Fill dataset preview
        if (data.dataset_preview && data.dataset_preview.length > 0) {
            document.getElementById('previewCount').textContent = data.dataset_preview.length;
            document.getElementById('totalRows').textContent = data.total_rows;
            
            // Create table header
            const tableHead = document.getElementById('datasetTableHead');
            const headerRow = document.createElement('tr');
            const columns = Object.keys(data.dataset_preview[0]);
            
            // Add row number column
            const rowNumHeader = document.createElement('th');
            rowNumHeader.textContent = '#';
            rowNumHeader.style.width = '50px';
            headerRow.appendChild(rowNumHeader);
            
            columns.forEach(col => {
                const th = document.createElement('th');
                th.textContent = col;
                headerRow.appendChild(th);
            });
            tableHead.appendChild(headerRow);
            
            // Create table body
            const dataTableBody = document.getElementById('datasetTableBody');
            data.dataset_preview.forEach((row, index) => {
                const tr = document.createElement('tr');
                
                // Add row number
                const rowNumCell = document.createElement('td');
                rowNumCell.textContent = index + 1;
                rowNumCell.style.fontWeight = 'bold';
                rowNumCell.style.color = '#667eea';
                tr.appendChild(rowNumCell);
                
                columns.forEach(col => {
                    const td = document.createElement('td');
                    const value = row[col];
                    
                    // Handle different data types
                    if (value === null || value === undefined) {
                        td.textContent = 'N/A';
                        td.style.color = '#999';
                    } else if (typeof value === 'number') {
                        td.textContent = value.toFixed(2);
                    } else {
                        td.textContent = value;
                    }
                    
                    tr.appendChild(td);
                });
                dataTableBody.appendChild(tr);
            });
        }

        // Create chart
        const dataPoints = data.actual.map((actual, i) => ({
            x: actual,
            y: data.predicted[i]
        }));

        const minVal = Math.min(...data.actual, ...data.predicted);
        const maxVal = Math.max(...data.actual, ...data.predicted);

        const perfectLine = [
            {x: minVal, y: minVal},
            {x: maxVal, y: maxVal}
        ];

        const ctx = document.getElementById('regressionChart').getContext('2d');
        
        new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Student Predictions',
                        data: dataPoints,
                        backgroundColor: 'rgba(52, 152, 219, 0.6)',
                        borderColor: 'rgba(52, 152, 219, 1)',
                        borderWidth: 1,
                        pointRadius: 6,
                        pointHoverRadius: 8
                    },
                    {
                        label: 'Perfect Prediction (y=x)',
                        data: perfectLine,
                        type: 'line',
                        borderColor: 'rgba(231, 76, 60, 1)',
                        backgroundColor: 'transparent',
                        borderWidth: 3,
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            font: { size: 14 }
                        }
                    },
                    title: {
                        display: true,
                        text: `Linear Regression: Actual vs Predicted | R² = ${data.r2.toFixed(4)}`,
                        font: { size: 16, weight: 'bold' }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'Actual Performance Index',
                            font: { size: 14, weight: 'bold' }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Predicted Performance Index',
                            font: { size: 14, weight: 'bold' }
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>
"""

# Replace placeholder with actual data
html_content = html_template.replace('DATA_PLACEHOLDER', json.dumps(results))

# Save HTML file
with open('regression_result.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\nHTML file generated: 'regression_result.html'")
