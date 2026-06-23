import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os
import sys

def auto_eda(filepath, target_col=None):
    print(f"\n{'='*60}")
    print(f"🔍 AUTO EDA REPORT — {os.path.basename(filepath)}")
    print(f"{'='*60}\n")

    df = pd.read_csv(filepath)

    #Basic Info
    print("📋 DATASET OVERVIEW")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"  Duplicate rows: {df.duplicated().sum()}")

    #Missing Values
    missing = df.isnull().sum()
    if missing.any():
        print("\n⚠️  MISSING VALUES:")
        miss_df = pd.DataFrame({
            'Count': missing[missing > 0],
            '%': (missing[missing > 0] / len(df) * 100).round(1)
        })
        print(miss_df.to_string())

    #Column Types
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"\n📊 COLUMN TYPES:")
    print(f"  Numerical ({len(num_cols)}): {num_cols}")
    print(f"  Categorical ({len(cat_cols)}): {cat_cols}")

    #Numerical Summary
    if num_cols:
        print("\n📈 NUMERICAL STATISTICS:")
        print(df[num_cols].describe().round(2).to_string())

    #Outlier Detection (IQR method)
    print("\n🎯 OUTLIER DETECTION (IQR Method):")
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        if outliers > 0:
            print(f"  {col}: {outliers} outliers ({outliers/len(df)*100:.1f}%)")

    #Correlation Analysis
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        print("\n🔗 TOP CORRELATIONS:")
        corr_pairs = corr.unstack().drop_duplicates().sort_values(key=abs, ascending=False)
        corr_pairs = corr_pairs[corr_pairs != 1.0].head(10)
        print(corr_pairs.to_string())

    #Target Analysis
    if target_col and target_col in df.columns:
        print(f"\n🎯 TARGET COLUMN: {target_col}")
        if target_col in num_cols:
            print(df[target_col].describe())
            skew = df[target_col].skew()
            print(f"  Skewness: {skew:.3f}")
        else:
            print(df[target_col].value_counts())

    #Visualizations
    n_plots = min(len(num_cols), 6)
    if n_plots > 0:
        fig = plt.figure(figsize=(18, n_plots * 3))
        gs = gridspec.GridSpec(n_plots, 3, figure=fig)

        for i, col in enumerate(num_cols[:n_plots]):
            #Histogram
            ax1 = fig.add_subplot(gs[i, 0])
            df[col].hist(bins=30, ax=ax1, color='steelblue', edgecolor='white')
            ax1.set_title(f'{col} — Distribution')

            #Boxplot
            ax2 = fig.add_subplot(gs[i, 1])
            df.boxplot(column=col, ax=ax2)
            ax2.set_title(f'{col} — Boxplot')

            #Q-Q Plot
            ax3 = fig.add_subplot(gs[i, 2])
            stats.probplot(df[col].dropna(), dist="norm", plot=ax3)
            ax3.set_title(f'{col} — Q-Q Plot')

        plt.tight_layout()
        output_path = filepath.replace('.csv', '_eda.png')
        plt.savefig(output_path, dpi=80, bbox_inches='tight')
        print(f"\n✅ EDA plots saved: {output_path}")

    #Correlation Heatmap
    if len(num_cols) > 2:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[num_cols].corr()
        im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha='right')
        plt.yticks(range(len(num_cols)), num_cols)
        for i in range(len(num_cols)):
            for j in range(len(num_cols)):
                plt.text(j, i, f'{corr_matrix.iloc[i,j]:.2f}',
                        ha='center', va='center', fontsize=8)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        heatmap_path = filepath.replace('.csv', '_heatmap.png')
        plt.savefig(heatmap_path, dpi=80)
        print(f"✅ Heatmap saved: {heatmap_path}")

    print(f"\n{'='*60}")
    print("✅ EDA Complete!")
    print(f"{'='*60}\n")

#Usage
if __name__ == "__main__":
    if len(sys.argv) > 1:
        auto_eda(sys.argv[1], target_col=sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        # Demo with synthetic data
        df_demo = pd.DataFrame({
            'age': np.random.randint(18, 80, 500),
            'income': np.random.exponential(50000, 500),
            'credit_score': np.random.randint(300, 850, 500),
            'loan_amount': np.random.uniform(1000, 50000, 500),
            'employment_years': np.random.randint(0, 40, 500),
            'default': np.random.choice(['Yes', 'No'], 500, p=[0.15, 0.85])
        })
        demo_path = '/tmp/demo_data.csv'
        df_demo.to_csv(demo_path, index=False)
        auto_eda(demo_path, target_col='default')