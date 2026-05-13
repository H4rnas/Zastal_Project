import cv2
import numpy as np
from pathlib import Path

def pick_hsv(image_path):
    """Kliknij na piksel – dostaniesz jego wartości HSV"""
    frame = cv2.imread(str(image_path))
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h, s, v = int(hsv[y, x][0]), int(hsv[y, x][1]), int(hsv[y, x][2])  # ← int() naprawia overflow
            print(f"Kliknięto ({x}, {y}) → H={h}  S={s}  V={v}")
            print(f"  lower = np.array([{max(0, h - 10)}, {max(0, s - 40)}, {max(0, v - 40)}])")
            print(f"  upper = np.array([{min(179, h + 10)}, {min(255, s + 40)}, {min(255, v + 40)}])")
            print("─────────────────")

    cv2.imshow("Klikaj na boisko", frame)
    cv2.setMouseCallback("Klikaj na boisko", on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


BASE_DIR = Path(__file__).parent.parent

path = BASE_DIR / "Food" / "Zastal_2_klatki" / "Klatka_boisko.png"

pick_hsv(str(path))