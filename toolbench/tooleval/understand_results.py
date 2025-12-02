import pandas as pd

csv_file = "data/pass_rate_results/virtual_gemini_cot/G1_category_virtual_gemini_cot.csv"
df = pd.read_csv(csv_file)

print(df.head())