import pandas as pd

# Load your Excel file
file_path = 'Export (19).xlsx'  # replace with your file name
df = pd.read_excel(file_path)

# Identify relevant columns
problem_cols = [col for col in df.columns if 'Problem' in col or 'Sample' in col]
pph_cols = [col for col in df.columns if 'PPH' in col]

# Apply forced changes
for col in problem_cols:
    df[col] = df[col].apply(
        lambda x: x + 10 if pd.notna(x) and isinstance(x, (int, float)) else x
    )

for col in pph_cols:
    df[col] = df[col].apply(
        lambda x: x + 2 if pd.notna(x) and isinstance(x, (int, float)) else x
    )

# Save the modified file
output_path = 'Mock_Export_Adjusted_Forced.xlsx'
df.to_excel(output_path, index=False)

print(f"Modified file saved as {output_path}")
