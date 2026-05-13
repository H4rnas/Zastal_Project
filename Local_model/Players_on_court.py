import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── SZKIELET COCO ─────────────────────────────────────────────────────────────
SKELETON = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

SPINE_COLORS = {
    "head_top": (255,   0, 255),
    "c1_th7":   (255, 165,   0),
    "lumbar":   (  0,   0, 255),
}
SPINE_LABELS = {
    "head_top": "Czubek glowy",
    "c1_th7":   "C1-Th7",
    "lumbar":   "Th12/L1",
}

# ── MASKA BOISKA ──────────────────────────────────────────────
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

def is_on_court(box, mask):
    """Sprawdza czy stopy gracza (środek dolnej krawędzi boxa) są na boisku"""
    x1, y1, x2, y2 = map(int, box)
    foot_x = (x1 + x2) // 2
    foot_y = min(int(y2), mask.shape[0] - 1)
    foot_x = min(foot_x,  mask.shape[1] - 1)
    return mask[foot_y, foot_x] > 0


# ── ESTYMACJA PUNKTÓW KRĘGOSŁUPA ──────────────────────────────────────────────
def estimate_spine_points(kpts, conf_threshold=0.5):
    nose  = kpts[0]
    l_ear = kpts[3];  r_ear = kpts[4]
    l_sh  = kpts[5];  r_sh  = kpts[6]
    l_hip = kpts[11]; r_hip = kpts[12]

    points = {}

    # Czubek głowy
    if nose[2] > conf_threshold:
        if l_ear[2] > conf_threshold and r_ear[2] > conf_threshold:
            avg_ear_y = (float(l_ear[1]) + float(r_ear[1])) / 2
            offset = abs(float(nose[1]) - avg_ear_y) * 1.3
        elif l_ear[2] > conf_threshold:
            offset = abs(float(nose[1]) - float(l_ear[1])) * 1.3
        elif r_ear[2] > conf_threshold:
            offset = abs(float(nose[1]) - float(r_ear[1])) * 1.3
        elif l_sh[2] > conf_threshold and r_sh[2] > conf_threshold:
            mid_sh_y = (float(l_sh[1]) + float(r_sh[1])) / 2
            offset = abs(float(nose[1]) - mid_sh_y) * 0.35
        else:
            offset = 20
        points["head_top"] = np.array([float(nose[0]), float(nose[1]) - offset, float(nose[2])])

    # C1-Th7
    if l_sh[2] > conf_threshold and r_sh[2] > conf_threshold:
        mid_x    = (float(l_sh[0]) + float(r_sh[0])) / 2
        mid_y    = (float(l_sh[1]) + float(r_sh[1])) / 2
        sh_width = abs(float(r_sh[0]) - float(l_sh[0]))
        points["c1_th7"] = np.array([mid_x, mid_y - sh_width * 0.15, float(min(l_sh[2], r_sh[2]))])

    # Th12/L1
    if l_hip[2] > conf_threshold and r_hip[2] > conf_threshold:
        mid_x = (float(l_hip[0]) + float(r_hip[0])) / 2
        mid_y = (float(l_hip[1]) + float(r_hip[1])) / 2
        if "c1_th7" in points:
            offset = abs(points["c1_th7"][1] - mid_y) * 0.15
        else:
            offset = 20
        points["lumbar"] = np.array([mid_x, mid_y - offset, float(min(l_hip[2], r_hip[2]))])

    return points


# ── RYSOWANIE ─────────────────────────────────────────────────────────────────
def draw_results(frame, kpts, conf_threshold=0.5):
    for x, y, conf in kpts:
        if conf > conf_threshold:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    for a, b in SKELETON:
        xa, ya, ca = kpts[a]
        xb, yb, cb = kpts[b]
        if ca > conf_threshold and cb > conf_threshold:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (0, 200, 255), 2)

    spine = estimate_spine_points(kpts, conf_threshold)

    for name, pt in spine.items():
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(frame, (x, y), 8, SPINE_COLORS[name], -1)
        cv2.circle(frame, (x, y), 8, (255, 255, 255), 1)
        cv2.putText(frame, SPINE_LABELS[name], (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, SPINE_COLORS[name], 2)

    if len(spine) == 3:
        spine_pts = [spine["head_top"], spine["c1_th7"], spine["lumbar"]]
        for i in range(len(spine_pts) - 1):
            p1 = (int(spine_pts[i][0]),   int(spine_pts[i][1]))
            p2 = (int(spine_pts[i+1][0]), int(spine_pts[i+1][1]))
            cv2.line(frame, p1, p2, (255, 255, 0), 2)

    return frame


# ── KOLORY DLA ID ZAWODNIKÓW ──────────────────────────────────────────────────
def get_color(track_id):
    """Każdy zawodnik ma swój unikalny kolor na podstawie ID"""
    np.random.seed(int(track_id))
    return tuple(int(x) for x in np.random.randint(50, 255, 3))


# ── GŁÓWNA PĘTLA Z TRACKINGIEM ────────────────────────────────────────────────
def process_video(input_path, output_path):
    detector   = YOLO("yolov8x.pt")
    pose_model = YOLO("yolov8x-pose.pt")
    for m in [detector, pose_model]:
        m.to("cuda")

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Maska boiska raz przed pętlą
    ret, first_frame = cap.read()
    court_mask = detect_court_mask(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"📹 Wideo: {w}x{h} @ {fps:.0f}fps")

    PADDING   = 0.15
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── ETAP 1: detekcja + tracking ───────────────────────────────────────
        track_results = detector.track(
            frame,
            device="cuda",
            classes=[0],        # tylko osoby
            conf=0.4,
            iou=0.5,
            half=True,
            verbose=False,
            persist=True,       # ← kluczowe – pamięta ID między klatkami
            tracker="bytetrack.yaml"
        )

        if track_results[0].boxes.id is None:
            out.write(frame)
            frame_num += 1
            continue

        boxes      = track_results[0].boxes.xyxy.cpu().numpy()
        track_ids  = track_results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box

            # Filtruj – tylko gracze na boisku
            if not is_on_court(box, court_mask):
                continue

            color = get_color(track_id)

            bw = x2 - x1
            bh = y2 - y1
            x1p = max(0, int(x1 - bw * PADDING))
            y1p = max(0, int(y1 - bh * PADDING))
            x2p = min(w, int(x2 + bw * PADDING))
            y2p = min(h, int(y2 + bh * PADDING))

            # Box z kolorem przypisanym do ID
            cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), color, 2)

            # Etykieta z numerem ID
            cv2.rectangle(frame, (x1p, y1p - 25), (x1p + 60, y1p), color, -1)
            cv2.putText(frame, f"ID {track_id}", (x1p + 4, y1p - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # ── ETAP 2: pose na cropie ────────────────────────────────────────
            crop = frame[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            pose_results = pose_model(crop, device="cuda", imgsz=256,
                                      conf=0.3, half=True, verbose=False)
            if pose_results[0].keypoints is None:
                continue

            for kpts_obj in pose_results[0].keypoints:
                kpts = kpts_obj.data[0].cpu().numpy().copy()
                kpts[:, 0] += x1p
                kpts[:, 1] += y1p
                frame = draw_results(frame, kpts)

        out.write(frame)
        frame_num += 1
        if frame_num % 30 == 0:
            print(f"⏱ Klatka {frame_num} ({frame_num/fps:.1f}s)")

    cap.release()
    out.release()
    print(f"✅ Gotowe! Zapisano: {output_path}")


# ── URUCHOMIENIE ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent

input_path  = BASE_DIR / "Food"   / "IMG_3165.mov"
output_path = BASE_DIR / "Output" / "Patyczaki_boisko4.mp4"

process_video(input_path, output_path)