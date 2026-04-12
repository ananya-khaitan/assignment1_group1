import pandas as pd
import os

df = pd.read_csv('Data/aligned_dataset.csv', index_col=0)
preds = ['SP500', 'Shanghai_Index', 'Crude_Oil', 'VIX', 'US_Dollar_Index', 'Search_Index', 'News_Sentiment']
names = ['S\\&P 500', 'Shanghai Index', 'Crude Oil', 'VIX', 'US Dollar Index', 'Search Index', 'News Sentiment']
stats = df[preds].agg(['mean', 'max', 'min', 'std', 'count']).T

tex = "\\begin{tabular}{lrrrrr} \\toprule Predictor & Mean & Max & Min & Std Dev & Obs \\\\ \\midrule \n"
for i, (idx, row) in enumerate(stats.iterrows()):
    tex += f"{names[i]} & {row['mean']:.4f} & {row['max']:.4f} & {row['min']:.4f} & {row['std']:.4f} & {int(row['count'])} \\\\ \n"
tex += "\\bottomrule \\end{tabular}"

with open('Results/Table_2_Predictor_Stats.tex', 'w') as f:
    f.write(tex)
print("Generated Predictor Stats Table")
