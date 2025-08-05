import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit

# Step 1: Load all CSV files from data/processed
csv_files = glob.glob('..\data\processed\*.csv')

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Combine all data into one DataFrame
df = pd.concat(dfs, ignore_index=True)

# Step 2: Convert LapTime (string) to timedelta and then to seconds
df['LapTime'] = pd.to_timedelta(df['LapTime'])
df['LapTimeSec'] = df['LapTime'].dt.total_seconds()

# Step 3: Filter only accurate laps
df_clean = df[df['IsAccurate'] == True]

# Step 4: Filter only for MEDIUM compound (change label if necessary)
MEDIUM_label = 'MEDIUM'  # Change this to 'C4' or other label if needed
df_MEDIUM = df_clean[df_clean['Compound'] == MEDIUM_label]

# Step 5: Scatter Plot: LapTimeSec vs TyreLife for MEDIUM
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_MEDIUM, x='TyreLife', y='LapTimeSec', alpha=0.4, color='red')
plt.title(f'Lap Time (sec) vs Tyre Life for {MEDIUM_label} Compound (raw data)')
plt.xlabel('Tyre Life (laps)')
plt.ylabel('Lap Time (seconds)')

# Save raw scatter plot
plt.savefig(f"../graphs/sel_graphs/raw_laptime_vs_tyrelife_General{MEDIUM_label.lower()}(scatterplot).png", dpi=300, bbox_inches='tight')
plt.show()

# Step 6: Polynomial curve fitting (degree 2) for MEDIUM compound

def poly2(x, a, b, c):
    return a*x**2 + b*x + c

# Filter typical stint lengths
df_MEDIUM_filtered = df_MEDIUM[(df_MEDIUM['TyreLife'] >= 1) & (df_MEDIUM['TyreLife'] <= 20)]

# Fit the polynomial
popt, _ = curve_fit(poly2, df_MEDIUM_filtered['TyreLife'], df_MEDIUM_filtered['LapTimeSec'])

# Generate fit line
x_fit = np.linspace(0, df_MEDIUM_filtered['TyreLife'].max(), 100)
y_fit = poly2(x_fit, *popt)

# Plot fit
plt.figure(figsize=(12, 6))
plt.plot(x_fit, y_fit, label=f'{MEDIUM_label} fit: {popt[0]:.4f}x² + {popt[1]:.4f}x + {popt[2]:.2f}', color='red')
sns.scatterplot(data=df_MEDIUM_filtered, x='TyreLife', y='LapTimeSec', alpha=0.2, color='red', label='Raw Data')

plt.title(f'Polynomial Fit of Lap Time vs Tyre Life for {MEDIUM_label} Compound')
plt.xlabel('Tyre Life (laps)')
plt.ylabel('Lap Time (seconds)')
plt.legend()

# Save fit plot
plt.savefig(f"../graphs/sel_graphs/polyfit_laptime_vs_tyrelife_{MEDIUM_label.lower()}(curvefit).png", dpi=300, bbox_inches='tight')
plt.show()
