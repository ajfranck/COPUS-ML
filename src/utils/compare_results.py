import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

#Once the json is converted to csv, we can put it here and compare it with craney's spreadsheets

# CHANGE THESE FILES FOR COMPARISON
craney_excel_file = "craney.xlsx"
model_output_file = "model.xlsx"

# Load spreadsheets
craney_df = pd.read_excel(craney_excel_file)
model_df = pd.read_excel(model_output_file)

# Ensure both have same rows
if len(craney_df) != len(model_df):
    raise ValueError("Files must have the same number of rows for comparison.")

summary = []
discrepancies = []

# Collect global results
all_true = []
all_pred = []

for col in craney_df.columns:
    if col not in model_df.columns:
        print(f"Skipping {col}, not found in model output")
        continue

    y_true = craney_df[col].astype(int)
    y_pred = model_df[col].astype(int)

    # Add to global
    all_true.extend(y_true)
    all_pred.extend(y_pred)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    agreement = (y_true == y_pred).mean()

    summary.append({
        "Code": col,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Accuracy": acc,
        "AgreementRate": agreement
    })

    # Record discrepancies row by row
    for idx in range(len(y_true)):
        if y_true.iloc[idx] != y_pred.iloc[idx]:
            discrepancies.append({
                "Row": idx,
                "Code": col,
                "Craney": y_true.iloc[idx],
                "Model": y_pred.iloc[idx]
            })

# Global metrics across ALL codes
tn, fp, fn, tp = confusion_matrix(all_true, all_pred, labels=[0,1]).ravel()
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

# Save summary to CSV
summary_df = pd.DataFrame(summary)
summary_df.to_csv("comparison_summary.csv", index=False)

# Save discrepancies to CSV
discrepancy_df = pd.DataFrame(discrepancies)
discrepancy_df.to_csv("comparison_discrepancies.csv", index=False)

print("Comparison complete. Results saved to:")
print(" - comparison_summary.csv (metrics per code + overall)")
print(" - comparison_discrepancies.csv (row-by-row mismatches)")

