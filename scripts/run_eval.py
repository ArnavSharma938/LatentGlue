import os
import subprocess
import sys

DEFAULT_CHECKPOINT_REPO_ID = "ArnavSharma938/LatentGlue"

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root_dir)
    env = os.environ.copy()
    env.setdefault("LATENTGLUE_CHECKPOINT", DEFAULT_CHECKPOINT_REPO_ID)
    subprocess.run([sys.executable, "-m", "src.validation.full_eval"], check=True, env=env)

if __name__ == "__main__":
    main()
