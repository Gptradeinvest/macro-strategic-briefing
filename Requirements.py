import subprocess
import sys

DEPS = [
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "feedparser>=6.0.0",
    "beautifulsoup4>=4.12.0",
    "scikit-learn>=1.3.0",
    "numpy>=1.24.0",
]

def main():
    for dep in DEPS:
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])
    print("done")

if __name__ == "__main__":
    main()
