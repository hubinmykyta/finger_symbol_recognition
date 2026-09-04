import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finger_symbol_recognition.app import AirDrawingApp


def main():
    app = AirDrawingApp()
    app.run()


if __name__ == "__main__":
    main()
