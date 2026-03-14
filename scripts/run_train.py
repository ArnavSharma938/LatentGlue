import subprocess
import os
import sys

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root_dir)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    subprocess.run([sys.executable, "-m", "src.model.train"], check=True)

if __name__ == "__main__":
    main()
