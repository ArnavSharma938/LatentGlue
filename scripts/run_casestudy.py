import os
import subprocess
import sys

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root_dir)
    subprocess.run([sys.executable, "-m", "src.casestudy.inference"], check=True)

if __name__ == "__main__":
    main()
