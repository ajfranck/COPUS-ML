import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from pathlib import Path

# File paths
CRANEY_FILE_PATH = Path(r"F:\taiwan project\2-minute COPUS Codes for 1 November First Camera (1) .xlsx")
MODEL_FILE_PATH = Path(r"C:\Users\Brendan Ng\COPUS-ML-main\data\results\copus_matrix.xlsx")

# Validate files exist
if not CRANEY_FILE_PATH.exists():
    raise FileNotFoundError(f"Craney file not found: {CRANEY_FILE_PATH}")
if not MODEL_FILE_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_FILE_PATH}")

# Load spreadsheets - skip first 2 rows (Times and Codes headers)
print("Loading files...")
craney_df = pd.read_excel(CRANEY_FILE_PATH, skiprows=2)
model_df = pd.read_excel(MODEL_FILE_PATH, skiprows=2)

# Set first column as index (COPUS code names like L, Ind, WG, etc.)
craney_df = craney_df.set_index(craney_df.columns[0])
model_df = model_df.set_index(model_df.columns[0])

# Remove rows with NaN index and completely empty rows
craney_df = craney_df[craney_df.index.notna()]
model_df = model_df[model_df.index.notna()]
craney_df = craney_df.dropna(how='all')
model_df = model_df.dropna(how='all')

# Strip whitespace from index names
craney_df.index = craney_df.index.str.strip()
model_df.index = model_df.index.str.strip()

print(f"\nCraney file shape: {craney_df.shape} (rows, cols)")
print(f"Model file shape: {model_df.shape} (rows, cols)")
print(f"\nCraney codes: {list(craney_df.index)}")
print(f"Model codes: {list(model_df.index)}")

# Find common codes
common_codes = craney_df.index.intersection(model_df.index)
print(f"\nComparing {len(common_codes)} common codes across time intervals...")

summary = []
discrepancies = []

# Collect global results
all_true = []
all_pred = []

for code in common_codes:
    # Get the rows for this code from both dataframes
    craney_row = craney_df.loc[code]
    model_row = model_df.loc[code]
    
    # Find common columns (time intervals) - only numeric ones
    common_cols = []
    for col in craney_row.index:
        if col in model_row.index:
            # Check if both values can be converted to numeric
            try:
                pd.to_numeric(craney_row[col])
                pd.to_numeric(model_row[col])
                common_cols.append(col)
            except:
                pass
    
    if len(common_cols) == 0:
        print(f"Skipping {code}, no common numeric columns")
        continue
    
    # Convert to numeric and fill NaN with 0
    y_true = pd.to_numeric(craney_row[common_cols], errors='coerce').fillna(0).astype(int)
    y_pred = pd.to_numeric(model_row[common_cols], errors='coerce').fillna(0).astype(int)

    # Add to global
    all_true.extend(y_true)
    all_pred.extend(y_pred)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    agreement = (y_true == y_pred).mean()

    summary.append({
        "Code": code,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Accuracy": acc,
        "AgreementRate": agreement
    })

    # Record discrepancies time interval by time interval
    mismatches = y_true != y_pred
    for time_col in mismatches[mismatches].index:
        discrepancies.append({
            "Code": code,
            "TimeInterval": time_col,
            "Craney": y_true[time_col],
            "Model": y_pred[time_col]
        })

# Global metrics across ALL codes
tn, fp, fn, tp = confusion_matrix(all_true, all_pred, labels=[0, 1]).ravel()
overall_acc = accuracy_score(all_true, all_pred)
overall_agreement = (pd.Series(all_true) == pd.Series(all_pred)).mean()

summary.append({
    "Code": "OVERALL",
    "TP": tp,
    "FP": fp,
    "FN": fn,
    "TN": tn,
    "Accuracy": overall_acc,
    "AgreementRate": overall_agreement
})

# Save results in the same directory as model output
output_dir = MODEL_FILE_PATH.parent
summary_path = output_dir / "comparison_summary.csv"
discrepancy_path = output_dir / "comparison_discrepancies.csv"

# Save summary to CSV
summary_df = pd.DataFrame(summary)
summary_df.to_csv(summary_path, index=False)

# Save discrepancies to CSV
discrepancy_df = pd.DataFrame(discrepancies)
discrepancy_df.to_csv(discrepancy_path, index=False)

print("\nComparison complete. Results saved to:")
print(f" - {summary_path}")
print(f" - {discrepancy_path}")
print(f"\nOverall Accuracy: {overall_acc:.2%}")
print(f"Overall Agreement Rate: {overall_agreement:.2%}")
print(f"Total Discrepancies: {len(discrepancy_df)}")