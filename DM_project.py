import os
import warnings
import logging

import pandas as pd
import numpy as np
import arff as liac
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Turn off warnings for clean output
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

# Show basic info for a DataFrame: shape, missing values, and simple stats
def summarize_df(df, label):
    print(f"\n=== {label} DataFrame Summary ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("Missing values per column:")
    print(df.isnull().sum())

    # Numeric columns: print mean and median
    nums = df.select_dtypes(include=[np.number]).columns
    stats = pd.DataFrame({
        'Mean': df[nums].mean().round(3),
        'Median': df[nums].median().round(3)
    })
    print("\nNumeric Mean & Median:")
    print(stats.to_string())

    # Categorical columns: show value counts
    cats = df.select_dtypes(include=['object', 'category']).columns
    if len(cats):
        print("\nCategorical counts:")
        for c in cats:
            print(c)
            print(df[c].value_counts(), "\n")

# Load data from an ARFF file into a pandas DataFrame
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    with open(path, 'r') as fp:
        raw = liac.load(fp)
    cols = [a[0] for a in raw['attributes']]
    return pd.DataFrame(raw['data'], columns=cols)

# Read the dataset and show its summary
df = load_data(r"C:\Users\fladn\Desktop\dm\Project_dm\dataset.arff")
summarize_df(df, 'Original')

# Add noisy copies of each row and bootstrap samples
def augment_dataset(df, copies=5, noise_scale=0.01, bootstrap_frac=1.0, random_state=42):
    np.random.seed(random_state)
    nums = df.select_dtypes(include=[np.number]).columns
    synthetic = []
    for _, row in df.iterrows():
        for _ in range(copies):
            new = row.copy()
            # Add small gaussian noise to numeric columns
            new[nums] = new[nums] + np.random.normal(0, noise_scale, len(nums))
            synthetic.append(new)
    # Bootstrap sampling of original data
    df_noise = pd.DataFrame(synthetic, columns=df.columns)
    df_boot = df.sample(frac=bootstrap_frac, replace=True, random_state=random_state)
    return pd.concat([df, df_noise, df_boot]).reset_index(drop=True)

# Create augmented dataset and show its summary
df_aug = augment_dataset(df)
summarize_df(df_aug, 'Augmented')

# Plot histograms to compare original vs augmented distributions
nums = df.select_dtypes(include=[np.number]).columns.drop('Stress_Level')
rows = int(np.ceil(len(nums) / 3))
fig, axes = plt.subplots(rows, 3, figsize=(24, 8 * rows))
axes = axes.flatten()
for ax, feat in zip(axes, nums):
    ax.hist(df_aug[feat], bins=15, alpha=0.4, edgecolor='white', label='Augmented')
    ax.hist(df[feat], bins=15, alpha=0.7, edgecolor='white', label='Original')
    ax.set_title(feat, fontsize=16)
    ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.02, 1))
for ax in axes[len(nums):]:
    ax.remove()
plt.subplots_adjust(hspace=0.8, wspace=0.6, right=0.85)
plt.show()

# Bar chart of absolute correlation of numeric features with target
numeric_cols = df.select_dtypes(include=[np.number]).columns
# Compute |corr| with Stress_Level, exclude the target itself
discorr = df[numeric_cols].corr()['Stress_Level'].abs().drop('Stress_Level').sort_values(ascending=False)
fig2, ax2 = plt.subplots(figsize=(12, 7))
discorr.plot(kind='bar', ax=ax2, edgecolor='black')
ax2.set_title('Feature Importance (|corr| with Stress_Level)', fontsize=16)
ax2.set_ylabel('Absolute Correlation', fontsize=14)
ax2.tick_params(labelsize=12)
plt.tight_layout(pad=2.0)
plt.show()

# Prepare data: scale numeric, encode categorical, then split
def prepare_data(df):
    X = df.drop(columns=['Stress_Level'])
    y = df['Stress_Level']
    pre = ColumnTransformer([
        ('scale', StandardScaler(), X.select_dtypes(include=[np.number]).columns),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False), X.select_dtypes(include=['object', 'category']).columns)
    ])
    X_pre = pre.fit_transform(X)
    return train_test_split(X_pre, y, test_size=0.2, random_state=42)

# Split original and augmented data
X_tr_o, X_te_o, y_tr_o, y_te_o = prepare_data(df)
X_tr_a, X_te_a, y_tr_a, y_te_a = prepare_data(df_aug)

# Define models to test
models = {
    'RF': RandomForestRegressor(random_state=42),
    'ExtraTrees': ExtraTreesRegressor(random_state=42),
    'GBR': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor()
}

# Compare R2 scores on original vs augmented data
perf = []
for name, mdl in models.items():
    # Train on original
    m0 = mdl.__class__(**mdl.get_params())
    m0.fit(X_tr_o, y_tr_o)
    r2o = r2_score(y_te_o, m0.predict(X_te_o))
    # Train on augmented
    m1 = mdl.__class__(**mdl.get_params())
    m1.fit(X_tr_a, y_tr_a)
    r2a = r2_score(y_te_a, m1.predict(X_te_a))
    perf.append((name, r2o, r2a))
perf_df = pd.DataFrame(perf, columns=['Model', 'R2_Orig', 'R2_Aug']).set_index('Model')
print("\n=== Model R² Comparison ===")
print(perf_df.round(3))

# Show bar chart of R2 scores
fig3, ax3 = plt.subplots(figsize=(12, 7))
perf_df.plot.bar(ax=ax3, width=0.7)
ax3.set_title('Model: Original vs Augmented R²', fontsize=16)
ax3.set_ylabel('R²', fontsize=14)
ax3.tick_params(labelsize=12)
plt.xticks(rotation=0)
plt.tight_layout(pad=2.0)
plt.show()

# Scatter plots of true vs predicted stress for each model
fig4, axes4 = plt.subplots(2, 3, figsize=(18, 14))
axes4 = axes4.flatten()
for ax, (name, mdl) in zip(axes4, models.items()):
    # Predictions
    m0 = mdl.__class__(**mdl.get_params()); m0.fit(X_tr_o, y_tr_o); p0 = m0.predict(X_te_o)
    m1 = mdl.__class__(**mdl.get_params()); m1.fit(X_tr_a, y_tr_a); p1 = m1.predict(X_te_a)
    ax.scatter(y_te_a, p1, alpha=0.4, s=80, label='Augmented')
    ax.scatter(y_te_o, p0, alpha=0.8, s=80, label='Original')
    mn = min(y_te_o.min(), y_te_a.min())
    mx = max(y_te_o.max(), y_te_a.max())
    ax.plot([mn, mx], [mn, mx], 'k--', linewidth=2)
    ax.set_title(name, fontsize=14)
    ax.set_xlabel('True Stress', fontsize=12)
    ax.set_ylabel('Predicted Stress', fontsize=12)
    ax.legend(fontsize=12)
for ax in axes4[len(models):]:
    ax.remove()
plt.subplots_adjust(hspace=0.6, wspace=0.5)
plt.show()
