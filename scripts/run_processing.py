import subprocess
import os
import sys

def run_step(command, description):
    print(f"\n[START] {description}...")
    try:
        subprocess.run([sys.executable] + command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {description} failed with exit code {e.returncode}")
        sys.exit(1)

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root_dir)

    run_step(["-m", "src.processing.SubSet"], "Step 1/4: Processing Raw Train data")

    run_step(["-m", "src.processing.EvalSet"], "Step 2/4: Processing Evaluation Data")

    run_step(["-m", "src.processing.ActivitySet"], "Step 3/4: Processing Activity Data")

    run_step(["-m", "src.processing.TrainSet"], "Step 4/4: Finalizing Training Data")

if __name__ == "__main__":
    main()
