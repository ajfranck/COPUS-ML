import json
import pandas as pd

JSON_FILE = r"c:/Users/zaned/COPUS-ML/data/results/json_files/test_lecture_copus.json"
OUT_XLSX  = r"c:/Users/zaned/COPUS-ML/data/results/copus_matrix.xlsx"

# --- COPUS code to full name ---
code_names = {
    "L": "Listening",
    "Ind": "Individual thinking",
    "CG": "Clicker question discussion",
    "WG": "Worksheet group work",
    "AnQ": "Answer instructor question",
    "SQ": "Student asks a question",
    "Lec": "Lecturing",
    "RtW": "Real-time writing",
    "FUp": "Follow-up",
    "PQ": "Pose questions",
    "CQ": "Clicker questions",
    "MG": "Moving through the classroom",
    "1o1": "One-on-one with students",
    "Adm": "Administration"
}

# --- Load JSON ---
with open(JSON_FILE, "r") as f:
    data = json.load(f)

intervals = data.get("intervals", [])
records = []

# --- Extract all interval-action pairs (True → 1, False → 0) ---
for i in intervals:
    interval_num = i.get("interval_number")
    actions = i.get("actions", {})
    if isinstance(actions, dict):
        for action, val in actions.items():
            records.append({
                "Interval": interval_num,
                "Action Code": action,
                "Value": int(bool(val))
            })
    else:
        records.append({
            "Interval": interval_num,
            "Action Code": "actions",
            "Value": int(bool(actions))
        })

# --- Create DataFrame ---
df = pd.DataFrame(records)

# --- Pivot so intervals become columns, actions become rows ---
pivot_df = df.pivot_table(index="Action Code", columns="Interval", values="Value", fill_value=0)

# --- Add Sum and Percent columns ---
pivot_df["Sum"] = pivot_df.sum(axis=1)
pivot_df["Percent"] = (pivot_df["Sum"] / len(pivot_df.columns[:-1])) * 100


# --- Export to Excel ---
pivot_df.to_excel(OUT_XLSX)

print(f"Excel file created at: {OUT_XLSX}")
