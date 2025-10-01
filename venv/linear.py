import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import random

np.random.seed(42)
random.seed(42)

countries = ['USA', 'China', 'Japan', 'Germany', 'India', 'UK', 'France', 'Italy', 
            'Brazil', 'Canada', 'Russia', 'South Korea', 'Spain', 'Australia', 'Mexico']
years = [2020, 2021, 2022, 2023, 2024, 2025]

# Базовые GDP значения в 2020 (в триллионах USD)
base_gdp_values = {
    'USA': 20.95, 'China': 14.72, 'Japan': 4.94, 'Germany': 3.85, 'India': 3.18,
    'UK': 2.83, 'France': 2.63, 'Italy': 2.11, 'Brazil': 1.61, 'Canada': 1.64,
    'Russia': 1.48, 'South Korea': 1.81, 'Spain': 1.39, 'Australia': 1.55, 'Mexico': 1.29
}

years_list = []
gdp_list = []

for country in countries:
    base_gdp = base_gdp_values[country]
    growth_rate = random.uniform(0.02, 0.04)  # темп роста 2-4% в год
    
    for i, year in enumerate(years):
        # GDP с трендом роста и случайными колебаниями
        gdp = base_gdp * ((1 + growth_rate) ** i) * random.uniform(0.95, 1.05)
        
        # Учитываем влияние COVID-19
        if year == 2020:
            gdp *= random.uniform(0.92, 0.98)
        elif year == 2021:
            gdp *= random.uniform(1.02, 1.08)
            
        years_list.append(year)
        gdp_list.append(gdp)

# Конвертируем в numpy массивы
X = np.array(years_list).reshape(-1, 1)  # годы
y = np.array(gdp_list)  # GDP

print(f"{len(years_list)} Observations were created for {len(countries)} countries")
print(f"Period: {min(years_list)}-{max(years_list)}")

model = LinearRegression()
model.fit(X, y)

# Получаем коэффициенты
intercept = model.intercept_
coefficient = model.coef_[0]

# Предсказания
y_pred = model.predict(X)

# Метрики качества
r2 = r2_score(y, y_pred)

print(f"\nLinear Regression results:")
print(f"Equotion: GDP = {coefficient:.4f} × year + {intercept:.2f}")
print(f"R² = {r2:.4f}")
print(f"Average annual GDP growth: {coefficient:.4f} trillion USD")

plt.figure(figsize=(12, 8))

# Точки данных
plt.scatter(years_list, gdp_list, alpha=0.6, color='steelblue', s=50, 
           label='GDP data by country', edgecolor='white', linewidth=0.5)

# Линия регрессии
x_line = np.array([2020, 2025]).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot([2020, 2025], y_line, 'red', linewidth=3, 
         label=f'Linear regression\nGDP = {coefficient:.4f}×year + {intercept:.2f}')

plt.xlabel('Year', fontsize=14)
plt.ylabel('GDP (trillion USD)', fontsize=14)
plt.title('Linear regression: GDP of world countries 2020-2025', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
         fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

print(f"\nInterpretation:")
print(f"The model shows that GDP grows on average of {coefficient:.4f} trillions USD every year")
print(f"R² = {r2:.4f} means that {r2*100:.1f}% GDP changes can be explained by the tempopary trend")