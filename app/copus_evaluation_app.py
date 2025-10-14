import os
import sys
import json
import argparse
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd
from xlsxwriter.utility import xl_rowcol_to_cell

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from lecture_evaluation.full_lecture_evaluation import FullLectureEvaluator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class COPUSEvaluationApp:
    """Main application class for COPUS video evaluation"""

    CODE_NAMES = {
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
        "TO": "Teacher other",
    }

    JSON_KEY_TO_CODE = {
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

    PREFERRED_ORDER = [
        # Students first
        "L",
        "Ind",
        "CG",
        "WG",
        "OG",
        "Prd",
        "TQ",
        "SAnQ",
        "SQ",
        "WC",
        "SP",
        "SW",
        "SO",
        # Instructors next
        "Lec",
        "RtW",
        "DV",
        "FUp",
        "PQ",
        "CQ",
        "TAnQ",
        "MG",
        "OoO",
        "Adm",
        "TW",
        "TO",
    ]

    def __init__(self, model_checkpoint=None, device="cuda"):
        self.model_checkpoint = model_checkpoint
        self.device = device
        self.evaluator = None

    def initialize_evaluator(self):
        logger.info("Intializing the evaluator...")
        self.evaluator = FullLectureEvaluator(
            checkpoint_path=self.model_checkpoint, device=self.device
        )
        logger.info("Evaluator initialized")

    def evaluate_video(self, video_path, output_dir):
        """
        Args:
            video_path: Path to the video file
            output_dir: Directory to save output files

        Returns:
            Tuple of (json_path, excel_path)
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        video_stem = video_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = output_dir / f"{video_stem}_copus_{timestamp}.json"
        excel_path = output_dir / f"{video_stem}_copus_{timestamp}.xlsx"

        logger.info(f"\n{'='*60}")
        logger.info(f"COPUS VIDEO EVALUATION")
        logger.info(f"{'='*60}")
        logger.info(f"Video: {video_path.name}")
        logger.info(f"Output direc: {output_dir}")
        logger.info(f"{'='*60}\n")

        if self.evaluator is None:
            self.initialize_evaluator()

        logger.info("Step 1/3: Evaluating")
        evaluation_results = self.evaluator.evaluate_full_lecture(
            str(video_path), str(json_path)
        )

        if "error" in evaluation_results:
            raise RuntimeError(f"evaluation fail: {evaluation_results['error']}")

        logger.info(f"JSON results: {json_path.name}")

        logger.info("\nStep 2/3: Convert to excel")
        self.convert_json_to_excel(json_path, excel_path)
        logger.info(f"excel: {excel_path.name}")

        logger.info("\nStep 3/3: Summary")
        self.generate_summary_report(
            evaluation_results, output_dir, video_stem, timestamp
        )

        return json_path, excel_path

    def convert_json_to_excel(self, json_path, excel_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        intervals = data.get("intervals", [])
        records = []

        for interval in intervals:
            interval_num = interval.get("interval_number")
            actions = interval.get("actions", {})

            if isinstance(actions, dict):
                for action, value in actions.items():
                    action_code = self.JSON_KEY_TO_CODE.get(action)
                    if action_code is None:
                        continue

                    records.append(
                        {
                            "Interval": interval_num,
                            "Action Code": action_code,
                            "Value": int(bool(value)),
                        }
                    )

        df = pd.DataFrame(records)
        if df.empty:
            logger.warning("No action data!!")
            return

        pivot_df = df.pivot_table(
            index="Action Code", columns="Interval", values="Value", fill_value=0
        )

        interval_cols = sorted(df["Interval"].unique())
        pivot_df = pivot_df.reindex(
            index=[
                c
                for c in self.PREFERRED_ORDER
                if c in pivot_df.index or c in self.JSON_KEY_TO_CODE.values()
            ],
            columns=interval_cols,
            fill_value=0,
        ).astype(int)

        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            workbook = writer.book
            worksheet = workbook.add_worksheet("COPUS_Analysis")
            writer.sheets["COPUS_Analysis"] = worksheet

            bold = workbook.add_format({"bold": True})
            center = workbook.add_format({"align": "center"})

            total_cols = 1 + len(interval_cols)
            video_name = Path(json_path).stem.replace("_copus", "").replace("_", " ")
            worksheet.merge_range(
                0, 0, 0, total_cols - 1, f"COPUS Analysis: {video_name}", bold
            )

            worksheet.write(1, 0, "2-min Intervals", bold)
            for idx, iv in enumerate(interval_cols):
                worksheet.write(1, 1 + idx, iv, center)

            col_sum = 1 + len(interval_cols)
            col_percent = col_sum + 1

            worksheet.write(1, col_sum, "Total", bold)
            worksheet.write(1, col_percent, "Percent", bold)

            worksheet.write(2, 0, "COPUS Codes", bold)

            code_to_row = {}
            for r, code in enumerate(
                [c for c in self.PREFERRED_ORDER if c in pivot_df.index]
            ):
                row_idx = 3 + r

                code_desc = f"{code} - {self.CODE_NAMES.get(code, code)}"
                worksheet.write(row_idx, 0, code_desc)

                if code in pivot_df.index:
                    row_vals = pivot_df.loc[code].tolist()
                else:
                    row_vals = [0] * len(interval_cols)

                for c, val in enumerate(row_vals):
                    worksheet.write(row_idx, 1 + c, int(val), center)

                code_to_row[code] = row_idx

                if interval_cols:
                    start_cell = xl_rowcol_to_cell(row_idx, 1)
                    end_cell = xl_rowcol_to_cell(row_idx, len(interval_cols))
                    worksheet.write_formula(
                        row_idx, col_sum, f"=SUM({start_cell}:{end_cell})"
                    )

                    sum_cell = xl_rowcol_to_cell(row_idx, col_sum)
                    worksheet.write_formula(
                        row_idx, col_percent, f"=({sum_cell}/{len(interval_cols)})*100"
                    )

            if code_to_row:
                summary_row = max(code_to_row.values()) + 3

                worksheet.write(summary_row, 0, "SUMMARY METRICS", bold)
                summary_row += 1

                categories = {
                    "Student Engagement": ["SAnQ", "SQ", "WC", "SP", "Ind", "WG", "OG"],
                    "Instructor Active": ["PQ", "FUp", "MG", "OoO", "TAnQ", "CQ"],
                    "Content Delivery": ["Lec", "RtW", "DV"],
                    "Passive Learning": ["L"],
                }

                for cat_name, codes in categories.items():
                    worksheet.write(summary_row, 0, cat_name)

                    refs = []
                    for code in codes:
                        if code in code_to_row:
                            refs.append(
                                xl_rowcol_to_cell(code_to_row[code], col_percent)
                            )

                    if refs:
                        formula = "=" + "+".join(refs)
                        worksheet.write_formula(summary_row, col_percent, formula)

                    summary_row += 1

            worksheet.set_column(0, 0, 30)
            worksheet.set_column(1, total_cols - 1, 8)
            worksheet.set_column(col_sum, col_percent, 12)

    def generate_summary_report(self, results, output_dir, video_stem, timestamp):
        """natural lang summary"""
        report_path = output_dir / f"{video_stem}_summary_{timestamp}.txt"

        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("COPUS VIDEO EVALUATION SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")

            video_info = results.get("video_info", {})
            f.write(f"Video: {Path(results['video_path']).name}\n")
            f.write(f"Duration: {video_info.get('duration_minutes', 0):.1f} minutes\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"Number of 2-minute intervals analyzed: {len(results.get('intervals', []))}\n"
            )
            f.write("\n")

            action_counts = {}
            total_intervals = len(results.get("intervals", []))

            for interval in results.get("intervals", []):
                for action, present in interval.get("actions", {}).items():
                    if present:
                        if action not in action_counts:
                            action_counts[action] = 0
                        action_counts[action] += 1

            sorted_actions = sorted(
                action_counts.items(), key=lambda x: x[1], reverse=True
            )

            f.write("MOST FREQUENT ACTIVITIES:\n")
            f.write("-" * 40 + "\n")

            for action, count in sorted_actions[:10]:
                percentage = (
                    (count / total_intervals * 100) if total_intervals > 0 else 0
                )
                code = self.JSON_KEY_TO_CODE.get(action, "")
                name = self.CODE_NAMES.get(code, action)
                f.write(f"  {name:35} {percentage:5.1f}%\n")

            f.write("\n")

            f.write("TEACHING STYLE ANALYSIS:\n")
            f.write("-" * 40 + "\n")

            instructor_active = [
                "instructor_posing_question",
                "instructor_follow_up",
                "instructor_moving_guiding",
                "instructor_one_on_one",
                "instructor_answering_question",
            ]
            content_delivery = [
                "instructor_lecturing",
                "instructor_real_time_writing",
                "instructor_demo_video",
            ]
            student_active = [
                "student_answer_question",
                "student_ask_question",
                "student_whole_class_discussion",
                "student_presentation",
                "student_individual_thinking",
                "student_worksheet_group",
                "student_other_group",
            ]

            categories = {
                "Interactive Teaching": instructor_active,
                "Content Delivery": content_delivery,
                "Student Engagement": student_active,
                "Passive Learning": ["student_listening"],
            }

            for cat_name, actions in categories.items():
                cat_count = sum(action_counts.get(a, 0) for a in actions)
                cat_percentage = (
                    (cat_count / total_intervals * 100) if total_intervals > 0 else 0
                )
                f.write(f"  {cat_name:25} {cat_percentage:5.1f}%\n")

            f.write("\n")

            f.write("ACTIVITY TIMELINE (First 10 intervals):\n")
            f.write("-" * 40 + "\n")

            for interval in results.get("intervals", [])[:10]:
                active_actions = [
                    a for a, p in interval.get("actions", {}).items() if p
                ]
                if active_actions:
                    top_actions = active_actions[:3]
                    action_codes = [
                        self.JSON_KEY_TO_CODE.get(a, "") for a in top_actions
                    ]
                    action_names = [
                        self.CODE_NAMES.get(c, c) for c in action_codes if c
                    ]

                    time_str = (
                        f"{interval['start_time_str']} - {interval['end_time_str']}"
                    )
                    actions_str = ", ".join(action_names[:3])
                    f.write(f"  {time_str:15} {actions_str}\n")

            f.write("\n")
            f.write("=" * 60 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 60 + "\n")

        logger.info(f"✓ Summary report saved: {report_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="COPUS Video Evaluation Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            %(prog)s lecture.mp4
            %(prog)s lecture.mp4 --output-dir results/
            %(prog)s lecture.mts --model-checkpoint models/copus_model_best/
            
            Output Files:
            - JSON file
            - Excel file
            - Text summary report
        """,
    )

    parser.add_argument(
        "video_file",
        type=str,
        help="Path to the video file to evaluate (MP4, MTS, AVI, MOV supported)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="copus_results",
        help="Directory to save output files (default: copus_results)",
    )

    parser.add_argument(
        "--model-checkpoint",
        "-m",
        type=str,
        default=None,
        help="Path to fine-tuned model checkpoint (optional)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference (default: cuda)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        app = COPUSEvaluationApp(
            model_checkpoint=args.model_checkpoint, device=args.device
        )

        json_path, excel_path = app.evaluate_video(args.video_file, args.output_dir)

        print("\n" + "=" * 60)
        print("Eval done")
        print("=" * 60)
        print(f"\nOutput files saved in: {args.output_dir}/")
        print(f"Excel: {excel_path.name}")
        print(f"JSON:  {json_path.name}")
        print(f"Summary:    {excel_path.stem.replace('_copus', '_summary')}.txt")
        print("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during eval: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
