import cv2
import numpy as np
from pathlib import Path

def detect_court_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Jasny parkiet – drewno (szeroki zakres S i V)
    lower_wood  = np.array([ 8,  50, 170])
    upper_wood  = np.array([28, 175, 255])

    # Zielona obwódka boiska
    lower_green = np.array([60,  80, 110])
    upper_green = np.array([95, 245, 155])

    mask_wood  = cv2.inRange(hsv, lower_wood,  upper_wood)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    mask = cv2.bitwise_or(mask_wood, mask_green)

    kernel = np.ones((30, 30), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # Zachowaj tylko największy obszar – boisko
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask_clean = np.zeros_like(mask)
        cv2.drawContours(mask_clean, [largest], -1, 255, thickness=cv2.FILLED)
        return mask_clean

    return mask

def preview_court_mask(image_path):
    """Podgląd maski na pojedynczej klatce"""
    frame = cv2.imread(str(image_path))
    mask  = detect_court_mask(frame)

    # Nałóż maskę na klatkę (boisko = kolorowe, reszta = ciemna)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Oryginał", frame)
    cv2.imshow("Maska boiska", mask)
    cv2.imshow("Wynik", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Użyj screenshota który wgrałeś

BASE_DIR = Path(__file__).parent.parent

path = BASE_DIR / "Food" / "Zastal_2_klatki" / "Klatka_boisko.png"

preview_court_mask(str(path))

