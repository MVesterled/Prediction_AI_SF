import cv2
import numpy as np
from pathlib import Path

# ---------- TUNE HERE IF NEEDED ----------
BASE_DIR = Path(__file__).resolve().parent  
# Dir with video
VIDEO_DIR = BASE_DIR / "images_vision"
# Image file
IMG_PATH = VIDEO_DIR / "CHE_vs_LIV4.png"
TARGET_SIZE = (1280, 720)

min_player_area = 40   # To count as a player
max_player_area = 12000

# ball radius fractions (of image height)
ball_rmin_frac = 0.006
ball_rmax_frac = 0.02

MIN_PITCH_OVERLAP = 0.60  # fraction of blob area that must lie inside pitch-region to count as player

# Arealvindue for ball mask konturer
BALL_MIN_AREA = 40    # px^2
BALL_MAX_AREA = 70    # px^2
# -----------------------------------------

# Pitch green (HSV) - mask
GREEN_LO = np.array([35,  35,  35])
GREEN_HI = np.array([90, 255, 255])

# Ball masks (White/orange)
BALL_WHITE_LO = np.array([0,   0, 127])
BALL_WHITE_HI = np.array([179, 73, 232])
BALL_ORANGE_LO = np.array([255, 255, 255])
BALL_ORANGE_HI = np.array([255, 255, 255])

# Teams – Liverpool red and Chelsea blue 
# red around 0-10 oandg 170-180 hue, high S to avoid skin
RED1_LO = np.array([0,   120, 80]);  RED1_HI = np.array([12, 255, 255])
RED2_LO = np.array([170, 120, 80]);  RED2_HI = np.array([179,255, 255])

# blue around 100-130 hue 
BLUE_LO = np.array([100, 120, 70]);  BLUE_HI = np.array([130,255,255])

# unwanted yellow (referee, ads)
YELLOW_LO = np.array([20, 120, 120]); YELLOW_HI = np.array([35, 255, 255])

# ---------- helpers ----------
def save_ball_possession_sentence(
    ball, bx, by, owner, red_pts, blue_pts,
    filename: str = "image_data.txt",
    overwrite: bool = True,
):
    # Ball position
    if ball is None:
        ball_txt = "the ball was not detected"
    else:
        ball_txt = f"the ball was found at pixel: (x={bx}, y={by})"

    # Ball possession
    if owner is None:
        poss_txt = "the ball is in possession of NONE"
    else:
        if owner in red_pts:
            who = "Liverpool"
        elif owner in blue_pts:
            who = "Chelsea"  # 
        else:
            who = "UNKNOWN"
        poss_txt = f"the ball is in possession of {who}"

    sentence = f"In this frame, {ball_txt}; {poss_txt}."

    # Write to file
    base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    out_dir = base_dir / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / filename

    mode = "w" if overwrite else "a"
    with target.open(mode, encoding="utf-8") as f:
        f.write(sentence + "\n")

    return str(target.resolve())

def clean(mask, k=3, it=1):
    kernel = np.ones((k,k), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=it)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=it)
    return m

def get_pitch_mask_and_region(hsv):
    pitch_mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    pitch_mask = clean(pitch_mask, 5, 1)
    region = cv2.dilate(pitch_mask, np.ones((21,21), np.uint8), iterations=1)
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8), iterations=1)
    return pitch_mask, region

# Ball detection med arealvindue
def detect_ball(hsv, pitch_mask, pitch_region, H):
    """
    Ball detection that only accepts candidates located on the field:
    - Color masks (white/orange) with yellow exclusion
    - Limited to the pitch area: intersection with pitch_region
    - Area window (BALL_MIN_AREA..BALL_MAX_AREA)
    - Post-check: radius and circularity
    - Overlap requirement with the pitch for each contour
    """

    # --- Colormasks ---
    white  = cv2.inRange(hsv, BALL_WHITE_LO,  BALL_WHITE_HI)
    orange = cv2.inRange(hsv, BALL_ORANGE_LO, BALL_ORANGE_HI)  # kan være tomt
    ball_mask = cv2.bitwise_or(white, orange)

    # Remove yellow (referee, ads)
    yellow = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    ball_mask = cv2.bitwise_and(ball_mask, cv2.bitwise_not(yellow))

    # *** only on the field ***
    ball_mask = cv2.bitwise_and(ball_mask, pitch_region)
    ball_mask = clean(ball_mask, 3, 1)

    # --- Filtration of area ---
    cnts_all, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Overlap with field requirement
    BALL_MIN_OVERLAP = 0.45  

    filtered = np.zeros_like(ball_mask)
    kept_cnts = []
    for c in cnts_all:
        a = cv2.contourArea(c)
        if a < BALL_MIN_AREA or a > BALL_MAX_AREA:
            continue

        # Overlap relative to the pitch_region on the contour’s ROI (Region of Interest)
        x, y, w, h = cv2.boundingRect(c)
        roi_region = pitch_region[y:y+h, x:x+w]
        roi_cnt = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(roi_cnt, [c - [x, y]], -1, 255, thickness=cv2.FILLED)

        overlap = cv2.countNonZero(cv2.bitwise_and(roi_cnt, roi_region))
        cnt_area = cv2.countNonZero(roi_cnt) + 1e-6
        if overlap / cnt_area < BALL_MIN_OVERLAP:
            continue

        kept_cnts.append(c)
        cv2.drawContours(filtered, [c], -1, 255, thickness=cv2.FILLED)

    ball_mask = filtered

    # --- radius and circularity ---
    rmin = int(ball_rmin_frac * H)
    rmax = int(ball_rmax_frac * H)

    best = None
    best_score = -1.0
    for c in kept_cnts:
        area = cv2.contourArea(c)
        if area < BALL_MIN_AREA or area > BALL_MAX_AREA:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4.0 * np.pi * area / (perim * perim)
        (x, y), r = cv2.minEnclosingCircle(c)
        r = int(r)

        if rmin <= r <= rmax and circularity > 0.6:
            if circularity > best_score:
                best_score = circularity
                best = (int(x), int(y), r)

    return best, ball_mask


def detect_players(hsv, pitch_region):
    # Liverpool red
    red1 = cv2.inRange(hsv, RED1_LO, RED1_HI)
    red2 = cv2.inRange(hsv, RED2_LO, RED2_HI)
    red = cv2.bitwise_or(red1, red2)
    red = clean(red, 5, 1)

    # Chelsea blå
    blue = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
    blue = clean(blue, 5, 1)

    def extract_centroids(mask):
        pts = []
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_player_area or a > max_player_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            roi_region = pitch_region[y:y+h, x:x+w]
            roi_cnt = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(roi_cnt, [c - [x, y]], -1, 255, thickness=cv2.FILLED)
            overlap = cv2.countNonZero(cv2.bitwise_and(roi_cnt, roi_region))
            cnt_area = cv2.countNonZero(roi_cnt) + 1e-6
            overlap_ratio = overlap / cnt_area
            if overlap_ratio < MIN_PITCH_OVERLAP:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            if pitch_region[cy, cx] == 0:
                continue
            pts.append((cx, cy))
        return pts

    red_pts  = extract_centroids(red)
    blue_pts = extract_centroids(blue)
    return red_pts, blue_pts, red, blue

def nearest_player(ball_xy, players, max_dist):
    if ball_xy is None or len(players)==0: return None, None
    bx, by = ball_xy
    best = None; dmin = 1e9
    for p in players:
        d = ((p[0]-bx)**2 + (p[1]-by)**2)**0.5
        if d < dmin: dmin, best = d, p
    if dmin <= max_dist: return best, dmin
    return None, None

# ############## MAIN ##############
# Load image
img = cv2.imread(IMG_PATH)
if img is None:
    raise SystemExit("Could not load image - check path.")

# Resize and convert to HSV
resized = cv2.resize(img, TARGET_SIZE)
draw = resized.copy()
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
H, W = resized.shape[:2]

# Build both masks
pitch_mask, pitch_region = get_pitch_mask_and_region(hsv)

# Find ball
ball, ball_mask = detect_ball(hsv, pitch_mask, pitch_region, H)
if ball is not None:
    bx, by, br = ball
    cv2.circle(draw, (bx,by), br, (255, 0, 255), 2)
    cv2.circle(draw, (bx,by), 3,  (255, 0, 255), -1)
else:
    bx = by = None

# Players
red_pts, blue_pts, red_mask, blue_mask = detect_players(hsv, pitch_region)
for (x,y) in red_pts:  cv2.circle(draw, (x,y), 6, (0,0,255), -1)   # Liverpool
for (x,y) in blue_pts: cv2.circle(draw, (x,y), 6, (255,0,0), -1)   # Chelsea

# Possession (nærmeste spiller til bolden)
owner = None; possession_txt = "Possession: NONE"
if bx is not None:
    max_d = 0.10 * (H**2 + W**2)**0.5
    r_near, dr = nearest_player((bx,by), red_pts+blue_pts, max_d)
    owner = r_near
    if owner is not None:
        cv2.line(draw, (bx,by), owner, (50,220,50), 2)
        cv2.circle(draw, owner, 10, (50,220,50), 2)
        possession_txt = "Possession: Liverpool" if owner in red_pts else "Possession: Chelsea"
else:
    possession_txt = "Ball not found"

cv2.putText(draw, possession_txt, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

# Terminal output
if ball is None:
    print("Ball: NOT FOUND")
else:
    print(f"Ball: FOUND at (x={bx}, y={by}), r={br}")
print(f"Liverpool RED (on pitch):  {len(red_pts)} -> {red_pts}")
print(f"Chelsea BLUE (on pitch):   {len(blue_pts)} -> {blue_pts}")
if owner is None:
    print("Possession: NONE")
else:
    who = "Liverpool" if owner in red_pts else "Chealsea"
    print(f"Possession: {who} (nearest player {owner})")

# Debug views
# cv2.imshow("ball mask", ball_mask)
# cv2.imshow("red mask", red_mask)
# cv2.imshow("blue mask", blue_mask)
# cv2.imshow("Possession", draw)
# cv2.waitKey(0); cv2.destroyAllWindows()

save_ball_possession_sentence(ball, bx, by, owner, red_pts, blue_pts)

