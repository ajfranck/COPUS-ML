import os
import sys
import json
import threading
import traceback
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

try:
    from copus_evaluation_app import COPUSEvaluationApp
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from copus_evaluation_app import COPUSEvaluationApp


class COPUSEvaluationGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("COPUS Video Evaluation System")
        self.root.geometry("900x700")

        style = ttk.Style()
        style.theme_use("clam")

        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="copus_results")
        self.model_checkpoint = tk.StringVar()
        self.device = tk.StringVar(value="cuda")
        self.processing = False

        self.create_widgets()

        self.center_window()

    def center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def create_widgets(self):

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        title_label = ttk.Label(
            main_frame, text="COPUS Video Evaluation System", font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        desc_text = (
            "Evaluate classroom videos using the COPUS (Classroom Observation Protocol "
            "for Undergraduate STEM) methodology. This tool generates comprehensive "
            "reports in Excel and JSON formats."
        )
        desc_label = ttk.Label(
            main_frame, text=desc_text, wraplength=850, justify=tk.LEFT
        )
        desc_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))

        ttk.Separator(main_frame, orient="horizontal").grid(
            row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )

        input_frame = ttk.LabelFrame(
            main_frame, text="Input Configuration", padding="10"
        )
        input_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Video File:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(input_frame, textvariable=self.video_path, width=60).grid(
            row=0, column=1, sticky=(tk.W, tk.E), pady=5
        )
        ttk.Button(input_frame, text="Browse...", command=self.browse_video).grid(
            row=0, column=2, padx=(5, 0), pady=5
        )

        ttk.Label(input_frame, text="Output Directory:").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(input_frame, textvariable=self.output_dir, width=60).grid(
            row=1, column=1, sticky=(tk.W, tk.E), pady=5
        )
        ttk.Button(input_frame, text="Browse...", command=self.browse_output_dir).grid(
            row=1, column=2, padx=(5, 0), pady=5
        )

        optional_frame = ttk.LabelFrame(
            main_frame, text="Optional Settings", padding="10"
        )
        optional_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        optional_frame.columnconfigure(1, weight=1)

        ttk.Label(optional_frame, text="Model Checkpoint:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(optional_frame, textvariable=self.model_checkpoint, width=60).grid(
            row=0, column=1, sticky=(tk.W, tk.E), pady=5
        )
        ttk.Button(
            optional_frame, text="Browse...", command=self.browse_checkpoint
        ).grid(row=0, column=2, padx=(5, 0), pady=5)

        ttk.Label(optional_frame, text="Processing Device:").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        device_frame = ttk.Frame(optional_frame)
        device_frame.grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(
            device_frame, text="CUDA (GPU)", variable=self.device, value="cuda"
        ).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(
            device_frame, text="CPU", variable=self.device, value="cpu"
        ).pack(side=tk.LEFT)

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=20)

        self.process_button = ttk.Button(
            button_frame,
            text="Start Evaluation",
            command=self.start_evaluation,
            style="Accent.TButton",
        )
        self.process_button.pack(side=tk.LEFT, padx=5)

        self.cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel_evaluation,
            state=tk.DISABLED,
        )
        self.cancel_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Clear Log", command=self.clear_log).pack(
            side=tk.LEFT, padx=5
        )

        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(
            side=tk.LEFT, padx=5
        )

        self.progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.progress.grid(
            row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="10")
        log_frame.grid(
            row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10
        )
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(7, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=15, width=100, wrap=tk.WORD
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.log("COPUS Video Evaluation System Ready")
        self.log("=" * 60)
        self.log("Please select a video file to begin evaluation.")

    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mts *.MTS"),
                ("MP4 Files", "*.mp4"),
                ("MTS Files", "*.mts *.MTS"),
                ("All Files", "*.*"),
            ],
        )
        if filename:
            self.video_path.set(filename)
            self.log(f"Selected video: {Path(filename).name}")

    def browse_output_dir(self):
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_dir.set(dirname)
            self.log(f"Output directory: {dirname}")

    def browse_checkpoint(self):
        """Browse for model checkpoint"""
        dirname = filedialog.askdirectory(title="Select Model Checkpoint Directory")
        if dirname:
            self.model_checkpoint.set(dirname)
            self.log(f"Model checkpoint: {dirname}")

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)

    def start_evaluation(self):
        if not self.video_path.get():
            messagebox.showerror("Error", "Select a video file")
            return

        if not Path(self.video_path.get()).exists():
            messagebox.showerror("Error", "Video file does not exist")
            return

        self.process_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.progress.start(10)
        self.processing = True

        self.clear_log()
        self.log("Starting evaluation.")
        self.log("=" * 60)

        # start processing in background thread
        thread = threading.Thread(target=self.run_evaluation, daemon=True)
        thread.start()

    def run_evaluation(self):
        """Run eval (in background thread)"""
        try:
            video_file = self.video_path.get()
            output_dir = self.output_dir.get()
            checkpoint = (
                self.model_checkpoint.get() if self.model_checkpoint.get() else None
            )
            device = self.device.get()

            self.log(f"Video: {Path(video_file).name}")
            self.log(f"Output directory: {output_dir}")
            self.log(f"Device: {device}")
            if checkpoint:
                self.log(f"Using checkpoint: {checkpoint}")
            self.log("")

            self.log("Initializing evaluator.")
            app = COPUSEvaluationApp(model_checkpoint=checkpoint, device=device)

            import logging

            class GUILogHandler(logging.Handler):
                def __init__(self, gui):
                    super().__init__()
                    self.gui = gui

                def emit(self, record):
                    msg = self.format(record)
                    self.gui.root.after(0, lambda: self.gui.log(msg))

            gui_handler = GUILogHandler(self)
            gui_handler.setFormatter(logging.Formatter("%(message)s"))
            logging.getLogger().addHandler(gui_handler)

            self.log("Starting video evaluation.")
            self.log("May take several minutes depending on video length.")

            json_path, excel_path = app.evaluate_video(video_file, output_dir)

            self.log("")
            self.log("=" * 60)
            self.log("Eval done")
            self.log("=" * 60)
            self.log("")
            self.log(f"Output files saved in: {output_dir}/")
            self.log(f"Excel: {excel_path.name}")
            self.log(f"JSON:  {json_path.name}")
            self.log(f"Summary:    {excel_path.stem.replace('_copus', '_summary')}.txt")

            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Success",
                    f"\n\nFiles saved in:\n{output_dir}\n\n"
                    f"• Excel: {excel_path.name}\n"
                    f"• JSON: {json_path.name}\n"
                    f"• Summary: {excel_path.stem.replace('_copus', '_summary')}.txt",
                ),
            )

            self.root.after(0, lambda: self.open_output_directory(output_dir))

        except Exception as e:
            error_msg = f"Error during evaluation: {str(e)}"
            self.log(f"\n{error_msg}")
            self.log("\nDetailed error:")
            self.log(traceback.format_exc())

            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

        finally:
            self.root.after(0, self.evaluation_complete)

    def open_output_directory(self, directory):
        if messagebox.askyesno("Open Output", "Open the output directory?"):
            import platform
            import subprocess

            system = platform.system()
            if system == "Windows":
                os.startfile(directory)
            elif system == "Darwin":  # mac
                subprocess.Popen(["open", directory])
            else:  # arch btw
                subprocess.Popen(["xdg-open", directory])

    def cancel_evaluation(self):
        self.processing = False
        self.log("\n user cancelled")
        self.evaluation_complete()

    def evaluation_complete(self):
        self.process_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        self.progress.stop()
        self.processing = False


def main():
    root = tk.Tk()
    app = COPUSEvaluationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
