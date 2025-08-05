import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit

# Step 1: Load all CSV files from data/processed
csv_files = glob.glob('..\data\processed\selective_processed\*.csv')

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

# Step 4: Plot raw data: LapTimeSec vs TyreLife by Compound
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_clean, x='TyreLife', y='LapTimeSec', hue='Compound', alpha=0.4)
plt.title('Lap Time (sec) vs Tyre Life by Compound (raw data)')
plt.xlabel('Tyre Life (laps)')
plt.ylabel('Lap Time (seconds)')
plt.legend(title='Compound')

# Save raw scatter plot
plt.savefig("../graphs/sel_graphs/raw_laptime_vs_tyrelife(scatterplot).png", dpi=300, bbox_inches='tight')


plt.show()


# Step 5: Polynomial curve fitting (degree 2) per compound

def poly2(x, a, b, c):
    return a*x**2 + b*x + c

plt.figure(figsize=(12, 6))
compounds = df_clean['Compound'].unique()
x_fit = np.linspace(0, df_clean['TyreLife'].max(), 100)

for comp in compounds:
    df_comp = df_clean[df_clean['Compound'] == comp]
    # Filter typical stint lengths
    df_comp_filtered = df_comp[(df_comp['TyreLife'] >= 1) & (df_comp['TyreLife'] <= 20)]
    
    popt, _ = curve_fit(poly2, df_comp_filtered['TyreLife'], df_comp_filtered['LapTimeSec'])
    y_fit = poly2(x_fit, *popt)
    plt.plot(x_fit, y_fit, label=f'{comp} fit: {popt[0]:.4f}x² + {popt[1]:.4f}x + {popt[2]:.2f}')

# Scatter raw data lightly for reference
sns.scatterplot(data=df_clean, x='TyreLife', y='LapTimeSec', hue='Compound', alpha=0.1, legend=False)

plt.title('Polynomial Fit of Lap Time vs Tyre Life by Compound')
plt.xlabel('Tyre Life (laps)')
plt.ylabel('Lap Time (seconds)')
plt.legend()

# Save polynomial fit plot
plt.savefig("../graphs/sel_graphs/polyfit_laptime_vs_tyrelife(curvefit).png", dpi=300, bbox_inches='tight')

plt.show()




