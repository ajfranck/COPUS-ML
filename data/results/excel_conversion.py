import json
import pandas as pd
from pathlib import Path
from xlsxwriter.utility import xl_rowcol_to_cell

# --- Paths relative to this script ---
BASE_DIR = Path(__file__).resolve().parent
JSON_FILE = BASE_DIR / "json_files" / "test_lecture_copus.json"
OUT_XLSX = BASE_DIR / "copus_matrix.xlsx"
# --- COPUS code to full name ---
code_names = {
    # Student codes
    "L": "Listening",
    "Ind": "Individual thinking",
    "CG": "Clicker question discussion",
    "WG": "Worksheet group work",
    "OG": "Other group work",
    "Prd": "Prediction",
    "TQ": "Test/quiz",
    "SAnQ": "Student answers a question",
    "SQ": "Student asks a question",
    "WC": "Whole-class discussion",
    "SP": "Student presentation",
    "SW": "Student waiting",
    "SO": "Student other",

    # Instructor codes
    "Lec": "Lecturing",
    "RtW": "Real-time writing",
    "FUp": "Follow-up",
    "PQ": "Pose questions",
    "CQ": "Clicker questions",
    "MG": "Moving through the classroom",
    "OoO": "One-on-one with students",
    "Adm": "Administration",
    "DV": "Demonstration/video/simulation",
    "TAnQ": "Teacher answers a question",
    "TW": "Teacher waiting",
}

# --- Map verbose JSON keys -> COPUS code ---
# Only keys present here will be included in the Excel output
json_key_to_code = {
    # Student activities
    "student_listening": "L",
    "student_individual_thinking": "Ind",
    "student_clicker_group": "CG",
    "student_worksheet_group": "WG",
    "student_other_group": "OG",
    "student_answer_question": "SAnQ",
    "student_ask_question": "SQ",
    "student_whole_class_discussion": "WC",
    "student_prediction": "Prd",
    "student_presentation": "SP",
    "student_test_quiz": "TQ",
    "student_waiting": "SW",
    "student_other": "SO",

    # Instructor activities
    "instructor_lecturing": "Lec",
    "instructor_real_time_writing": "RtW",
    "instructor_follow_up": "FUp",
    "instructor_posing_question": "PQ",
    "instructor_clicker_question": "CQ",
    "instructor_moving_guiding": "MG",
    "instructor_one_on_one": "OoO",
    "instructor_demo_video": "DV",
    "instructor_administration": "Adm",
    "instructor_waiting": "TW",
    "instructor_answering_question": "TAnQ",
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
            # Only include actions that map to a COPUS code
            action_code = json_key_to_code.get(action)
            if action_code is None:
                continue
            records.append({
                "Interval": interval_num,
                "Action Code": action_code,
                "Value": int(bool(val))
            })

# --- Create DataFrame and pivot ---
df = pd.DataFrame(records)
pivot_df = df.pivot_table(index="Action Code", columns="Interval", values="Value", fill_value=0)

# --- Exact display order (from your screenshot) ---
preferred_order = [
    # Students first
    "L", "Ind", "WG", "OG", "Prd", "TQ", "SAnQ", "SQ", "WC", "SP", "SW", "SO",
    # Instructors next
    "Lec", "RtW", "DV", "FUp", "PQ", "TAnQ", "MG", "OoO", "Adm", "TW", "TO",
]

# Ensure all desired rows/columns exist and are sorted
interval_cols = sorted(df["Interval"].unique())
pivot_df = pivot_df.reindex(index=preferred_order, columns=interval_cols, fill_value=0).astype(int)

# --- Export to Excel with top 3 rows ---
with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
    workbook = writer.book
    worksheet = workbook.add_worksheet("Sheet1")
    writer.sheets["Sheet1"] = worksheet

    bold = workbook.add_format({"bold": True})

    # Row 1: Title merged across all columns (A1 .. last)
    total_cols = 1 + len(interval_cols)  # left label + intervals
    worksheet.merge_range(0, 0, 0, total_cols - 1, JSON_FILE.name, bold)

    # Row 2: Times header
    worksheet.write(1, 0, "Times", bold)
    for idx, iv in enumerate(interval_cols):
        worksheet.write(1, 1 + idx, iv)

    # Append summary headers at end
    col_sum = 1 + len(interval_cols)
    col_percent = col_sum + 1
    col_student_talk = col_percent + 1
    col_instr_guide = col_student_talk + 1
    col_student_work = col_instr_guide + 1
    col_teach_present = col_student_work + 1
    col_student_rec = col_teach_present + 1
    col_adm_to = col_student_rec + 1
    col_take_quiz_sq = col_adm_to + 1

    worksheet.write(1, col_sum, "Time Sum", bold)
    worksheet.write(1, col_percent, "Time Percent", bold)
    worksheet.write(1, col_student_talk, "Student Talk", bold)
    worksheet.write(1, col_instr_guide, "Instructor Guide", bold)
    worksheet.write(1, col_student_work, "Student Work", bold)
    worksheet.write(1, col_teach_present, "Teach Present", bold)
    worksheet.write(1, col_student_rec, "Student Rec", bold)
    worksheet.write(1, col_adm_to, "Amd +TO", bold)
    worksheet.write(1, col_take_quiz_sq, "Take Quiz + SQ", bold)

    # Row 3: Codes header
    worksheet.write(2, 0, "Codes", bold)

    # Rows 4+: Codes and matrix
    # Build lookup: code -> data row index in worksheet (0-based)
    code_to_row = {}
    for r, code in enumerate(preferred_order):
        worksheet.write(3 + r, 0, code)
        if code in pivot_df.index:
            row_vals = pivot_df.loc[code].tolist()
        else:
            row_vals = [0] * len(interval_cols)
        for c, val in enumerate(row_vals):
            worksheet.write(3 + r, 1 + c, int(val))

        # Record row mapping
        code_to_row[code] = 3 + r

        # Time Sum formula across interval cells
        if interval_cols:
            start_cell = xl_rowcol_to_cell(3 + r, 1)
            end_cell = xl_rowcol_to_cell(3 + r, 1 + len(interval_cols) - 1)
            worksheet.write_formula(3 + r, col_sum, f"=SUM({start_cell}:{end_cell})")
            # Time Percent = Time Sum / total intervals * 100
            sum_cell = xl_rowcol_to_cell(3 + r, col_sum)
            worksheet.write_formula(3 + r, col_percent, f"=({sum_cell}/{len(interval_cols)})*100")

    # After rows are written, write composite formulas (reference Time Percent column for specific codes)
    def pct_cell_for(code: str) -> str:
        row = code_to_row.get(code)
        if row is None:
            return None
        return xl_rowcol_to_cell(row, col_percent)

    components = {
        "Student Talk": ["SAnQ", "SQ", "WC", "SP"],
        "Instructor Guide": ["PQ", "FUp", "MG", "OoO", "TAnQ"],
        "Student Work": ["Ind", "WG", "OG", "Prd"],
        "Teach Present": ["Lec", "RtW", "DV"],
        "Student Rec": ["L"],
        "Amd +TO": ["Adm", "TO"],
        "Take Quiz + SQ": ["TQ", "SQ"],
    }

    # Write composite formulas ONCE in a single summary row (use the first code row)
    summary_row = 3  # row index of the first code (e.g., 'L')
    # Student Talk
    refs = [pct_cell_for(c) for c in components["Student Talk"] if pct_cell_for(c)]
    if refs:
        worksheet.write_formula(summary_row, col_student_talk, "=" + "+".join(refs))
    # Instructor Guide
    refs = [pct_cell_for(c) for c in components["Instructor Guide"] if pct_cell_for(c)]
    if refs:
        worksheet.write_formula(summary_row, col_instr_guide, "=" + "+".join(refs))
    # Student Work
    refs = [pct_cell_for(c) for c in components["Student Work"] if pct_cell_for(c)]
    if refs:
        worksheet.write_formula(summary_row, col_student_work, "=" + "+".join(refs))
    # Teach Present
    refs = [pct_cell_for(c) for c in components["Teach Present"] if pct_cell_for(c)]
    if refs:
        worksheet.write_formula(summary_row, col_teach_present, "=" + "+".join(refs))
    # Student Rec
    refs = [pct_cell_for(c) for c in components["Student Rec"] if pct_cell_for(c)]
    if refs:
        worksheet.write_formula(summary_row, col_student_rec, "=" + "+".join(refs))
    # Amd + TO
    refs = [pct_cell_for(c) for c in components["Amd +TO"] if pct_cell_for(c)]
    if refs:
        worksheet.write_formula(summary_row, col_adm_to, "=" + "+".join(refs))
    # Take Quiz + SQ
    refs = [pct_cell_for(c) for c in components["Take Quiz + SQ"] if pct_cell_for(c)]
    if refs:
        worksheet.write_formula(summary_row, col_take_quiz_sq, "=" + "+".join(refs))

print(f"Excel file created at: {OUT_XLSX}")
