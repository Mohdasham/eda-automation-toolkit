# EDA Automation Toolkit

> Drop any CSV and instantly get a full exploratory data analysis report with visualizations.

## Features
- Missing value detection
- Outlier detection (IQR method)
- Correlation analysis
- Distribution plots, boxplots, Q-Q plots
- Automatic heatmap generation

## Compatible With Any CSV Dataset
Example: [YouTube Trending Dataset — Kaggle](https://www.kaggle.com/datasets/datasnaek/youtube-new)

## Tech Stack
- Python, Pandas, NumPy, Matplotlib, SciPy

## Quick Start
```bash
git clone git@github.com:Mohdasham/eda-automation-toolkit.git
cd eda-automation-toolkit
pip install -r requirements.txt

# Run on any CSV
python auto_eda.py your_data.csv target_column
```
