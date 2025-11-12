import cv2
import numpy as np
import os
from pathlib import Path

# ---------- TUNE HERE IF NEEDED ----------
BASE_DIR = Path(__file__).resolve().parent  # fallback til Path.cwd() hvis du kører i REPL
# Dir with video
VIDEO_DIR = BASE_DIR / "video_vision"
# Videofile
VIDEO_PATH = VIDEO_DIR / "CHE_VS_LIV_Video.mp4"

TARGET_SIZE = (1280, 720)

min_player_area = 100
max_player_area = 12000

# Ball size relative to image height (extra check)
ball_rmin_frac = 0.006
ball_rmax_frac = 0.020

MIN_PITCH_OVERLAP = 0.60   # fraction of blob area that must lie inside pitch-region to count as player

# Area window for ball-mask contours (px^2)
BALL_MIN_AREA = 40
BALL_MAX_AREA = 70

# Real-time tuning
FRAME_SKIP = 0              # e.g. set 1 to process every 2nd frame
SMOOTH_ALPHA = 0.2         # 0..1, lower gives more smoothing
KEEP_BALL_FOR = 15           # how many frames to keep last known ball if it is not seen

WRITE_OUT = True
OUT_PATH = os.path.splitext(VIDEO_PATH)[0] + "_tracked.mp4"

# NEW: anti-randomization controls
ROI_BASE   = 60             # px base radius for ROI when searching next ball
ROI_SCALE  = 6              # ROI radius ~= ROI_BASE or ROI_SCALE * last_ball_r (whichever larger)
JUMP_GATE  = 0.08           # fraction of image diagonal as max allowed jump per re-detection
SHOW_GHOST = False          # draw last known ball when missing (False = don't show)
# -----------------------------------------

# Pitch green (HSV)
GREEN_LO = np.array([35,  35,  35])
GREEN_HI = np.array([90, 255, 255])

# Ball masks (white/orange)
BALL_WHITE_LO = np.array([0,   0, 127])
BALL_WHITE_HI = np.array([179, 73, 232])
# Orange disabled (no interval)
BALL_ORANGE_LO = np.array([255, 255, 255])
BALL_ORANGE_HI = np.array([255, 255, 255])

# Teams – Liverpool red and Chelsea blue
RED1_LO = np.array([0,   120, 80]);  RED1_HI = np.array([12, 255, 255])
RED2_LO = np.array([170, 120, 80]);  RED2_HI = np.array([179,255, 255])
BLUE_LO = np.array([100, 120, 70]);  BLUE_HI = np.array([130,255,255])

# Optional: suppress referee yellow
YELLOW_LO = np.array([20, 120, 120]); YELLOW_HI = np.array([35, 255, 255])


# ---------- helpers ----------

def save_possession_summary(red_pct, blue_pct, overwrite=True):
    sentence = (
        f"In the video, Liverpool and Chelsea are playing. "
        f"Using ball detection, we can see that Liverpool has possession "
        f"for {red_pct:.1f}% of the time, while Chelsea has {blue_pct:.1f}%."
    )

    docs = Path(__file__).resolve().parent / "docs"
    docs.mkdir(exist_ok=True)
    mode = "w" if overwrite else "a"
    with (docs / "video_data.txt").open(mode, encoding="utf-8") as f:
        f.write(sentence + "\n")

def clean(mask, k=3, it=1):
    kernel = np.ones((k,k), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=it)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=it)
    return m

def get_pitch_mask_and_region(hsv):
    pitch_mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    pitch_mask = clean(pitch_mask, 5, 1)
    # Expand pitch and close holes/lines
    region = cv2.dilate(pitch_mask, np.ones((21,21), np.uint8), iterations=1)
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8), iterations=1)
    return pitch_mask, region

def nearest_player(ball_xy, players, max_dist):
    if ball_xy is None or len(players) == 0:
        return None, None
    bx, by = ball_xy
    best = None; dmin = 1e9
    for p in players:
        d = ((p[0] - bx)**2 + (p[1] - by)**2)**0.5
        if d < dmin:
            dmin, best = d, p
    if dmin <= max_dist:
        return best, dmin
    return None, None

def smooth_point(prev, new, alpha):
    if prev is None or new is None:
        return new
    x = int(alpha*new[0] + (1 - alpha)*prev[0])
    y = int(alpha*new[1] + (1 - alpha)*prev[1])
    return (x, y)


# ---------- detection ----------
def detect_ball(hsv, pitch_mask, pitch_region, H,
                players_mask=None, prev_xy=None, roi_mask=None,
                jump_gate_diag_frac=0.08):
    """
    Ball detection restricted to the pitch, with:
      - top-30% exclusion,
      - optional ROI,
      - player-exclusion (red|blue, dilated),
      - area window,
      - pitch-overlap,
      - shape gates (circularity + aspect ratio),
      - context ring test (must be mostly green, little team color),
      - proximity scoring vs prev_xy + jump gate.
    Returns: (x,y,r) or None, and the final ball_mask.
    """
    # --- Base white/orange mask ---
    white  = cv2.inRange(hsv, BALL_WHITE_LO,  BALL_WHITE_HI)
    orange = cv2.inRange(hsv, BALL_ORANGE_LO, BALL_ORANGE_HI)
    ball_mask = cv2.bitwise_or(white, orange)

    # Remove yellow
    yellow = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    ball_mask = cv2.bitwise_and(ball_mask, cv2.bitwise_not(yellow))

    # Only on pitch
    ball_mask = cv2.bitwise_and(ball_mask, pitch_region)

    # --- NEW: exclude (dilated) players to kill white socks close to kits ---
    if players_mask is not None:
        excl = cv2.dilate(players_mask, np.ones((9,9), np.uint8), iterations=1)
        ball_mask = cv2.bitwise_and(ball_mask, cv2.bitwise_not(excl))

    # Clean and hard cap top 30%
    ball_mask = clean(ball_mask, 3, 1)
    top_cut = int(0.30 * H)
    ball_mask[:top_cut, :] = 0

    # Optional ROI
    if roi_mask is not None:
        ball_mask = cv2.bitwise_and(ball_mask, roi_mask)

    # Contours
    cnts_all, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pitch-overlap filter
    BALL_MIN_OVERLAP = 0.50
    filtered = np.zeros_like(ball_mask)
    kept_cnts = []
    for c in cnts_all:
        a = cv2.contourArea(c)
        if a < BALL_MIN_AREA or a > BALL_MAX_AREA:
            continue

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

    # Final scoring (shape + context + proximity)
    rmin = int(ball_rmin_frac * H)
    rmax = int(ball_rmax_frac * H)
    diag = (hsv.shape[0]**2 + hsv.shape[1]**2)**0.5
    jump_gate = jump_gate_diag_frac * diag

    best = None
    best_score = -1.0

    # precompute masks we need for context
    pitch_only = (pitch_region > 0).astype(np.uint8)
    red_mask   = cv2.inRange(hsv, RED1_LO, RED1_HI) | cv2.inRange(hsv, RED2_LO, RED2_HI)
    blue_mask  = cv2.inRange(hsv, BLUE_LO, BLUE_HI)

    for c in kept_cnts:
        area = cv2.contourArea(c)
        if area < BALL_MIN_AREA or area > BALL_MAX_AREA:
            continue

        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circ = 4.0 * np.pi * area / (perim * perim)

        # aspect ratio via minAreaRect
        rect = cv2.minAreaRect(c)
        (w, h) = rect[1]
        if w == 0 or h == 0:
            continue
        ar = max(w, h) / max(1.0, min(w, h))  # >=1
        if ar > 1.8:             # socks are elongated -> reject
            continue

        (x, y), r = cv2.minEnclosingCircle(c)
        r = int(r)
        if not (rmin <= r <= rmax):
            continue

        # jump gate / proximity
        prox_term = 0.5
        if prev_xy is not None:
            d = ((x - prev_xy[0])**2 + (y - prev_xy[1])**2)**0.5
            if d > jump_gate:
                continue
            prox_term = max(0.0, 1.0 - (d / jump_gate))

        # --- Context ring test ---
        # annulus around candidate: [1.3r .. 2.2r]
        r1 = int(1.3 * r); r2 = int(2.2 * r)
        yy, xx = np.ogrid[:hsv.shape[0], :hsv.shape[1]]
        cy = int(y); cx = int(x)
        y0 = max(0, cy - r2); y1 = min(hsv.shape[0], cy + r2 + 1)
        x0 = max(0, cx - r2); x1 = min(hsv.shape[1], cx + r2 + 1)
        sub_pitch = pitch_only[y0:y1, x0:x1]
        sub_red   = red_mask[y0:y1, x0:x1]   > 0
        sub_blue  = blue_mask[y0:y1, x0:x1]  > 0

        yy2, xx2 = np.ogrid[y0:y1, x0:x1]
        ann = (xx2 - cx)**2 + (yy2 - cy)**2
        ring = (ann <= r2*r2) & (ann >= r1*r1)

        ring_count = np.count_nonzero(ring)
        if ring_count < 10:   # too small for a stable test
            continue

        green_in_ring = np.count_nonzero(ring & (sub_pitch > 0))
        team_in_ring  = np.count_nonzero(ring & (sub_red | sub_blue))
        green_ratio = green_in_ring / ring_count
        team_ratio  = team_in_ring  / ring_count

        if green_ratio < 0.35:   # not enough grass around -> likely sock/foot
            continue
        if team_ratio > 0.30:    # too much kit color around -> likely sock/foot
            continue

        # Score: circularity + proximity + green context
        circ_term = np.clip((circ - 0.35) / (0.80 - 0.35), 0.0, 1.0)
        ctx_term  = np.clip(green_ratio - 0.15, 0.0, 1.0)  # boost when more grass
        score = 0.5*circ_term + 0.3*prox_term + 0.2*ctx_term

        if score > best_score:
            best_score = score
            best = (int(x), int(y), r)

    return best, ball_mask



def detect_players(hsv, pitch_region):
    # Liverpool red
    red1 = cv2.inRange(hsv, RED1_LO, RED1_HI)
    red2 = cv2.inRange(hsv, RED2_LO, RED2_HI)
    red = cv2.bitwise_or(red1, red2)
    red = clean(red, 5, 1)

    # Chelsea blue
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
            if M["m00"] == 0:
                continue
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            if pitch_region[cy, cx] == 0:
                continue
            pts.append((cx, cy))
        return pts

    red_pts  = extract_centroids(red)
    blue_pts = extract_centroids(blue)
    return red_pts, blue_pts, red, blue


# ############## VIDEO LOOP ##############
# Possession counters
red_frames = 0
blue_frames = 0
none_frames = 0
frames_counted = 0

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
writer = None
if WRITE_OUT:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUT_PATH, fourcc, fps, TARGET_SIZE)

frame_idx = 0
last_ball_xy = None
last_ball_r = None
misses = 0
last_possession_str = "Possession: NONE"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if FRAME_SKIP and (frame_idx % (FRAME_SKIP+1) != 0):
        frame_idx += 1
        continue

    resized = cv2.resize(frame, TARGET_SIZE)
    draw = resized.copy()
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    H, W = resized.shape[:2]
    diag = (H**2 + W**2) ** 0.5

    pitch_mask, pitch_region = get_pitch_mask_and_region(hsv)

    # Build ROI mask from last known position
    roi_mask = None
    if last_ball_xy is not None:
        roi_r = int(max(ROI_BASE, ROI_SCALE * (last_ball_r if last_ball_r else 8)))
        roi_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(roi_mask, (int(last_ball_xy[0]), int(last_ball_xy[1])), roi_r, 255, -1)

    # Detect ball with ROI + jump-gate against last position
    # compute players first
    red_pts, blue_pts, red_mask, blue_mask = detect_players(hsv, pitch_region)
    players_mask = cv2.bitwise_or(red_mask, blue_mask)

    # optional ROI omkring sidste kendte bold
    roi_mask = None
    if last_ball_xy is not None:
        roi_r = int(max(60, 6 * (last_ball_r if last_ball_r else 8)))
        roi_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(roi_mask, (int(last_ball_xy[0]), int(last_ball_xy[1])), roi_r, 255, -1)

    # now detect the ball with player-exclusion + context + ROI
    ball, ball_mask = detect_ball(
    hsv, pitch_mask, pitch_region, H,
    players_mask=players_mask,
    prev_xy=last_ball_xy,
    roi_mask=roi_mask,
    jump_gate_diag_frac=0.08
)

    if ball is not None:
        bx, by, br = ball
        sm_xy = smooth_point(last_ball_xy, (bx, by), SMOOTH_ALPHA)
        sm_r = int(SMOOTH_ALPHA*br + (1-SMOOTH_ALPHA)*(last_ball_r if last_ball_r else br))
        last_ball_xy = sm_xy
        last_ball_r  = sm_r
        misses = 0
    else:
        misses += 1
        if misses <= KEEP_BALL_FOR and last_ball_xy is not None and SHOW_GHOST:
            bx, by = last_ball_xy
            br = last_ball_r if last_ball_r else int(ball_rmin_frac*H)
        else:
            bx = by = br = None
            last_ball_xy = None
            last_ball_r  = None

    # Draw ball if present
    if bx is not None:
        cv2.circle(draw, (bx, by), int(br), (255, 0, 255), 2)
        cv2.circle(draw, (bx, by), 3, (255, 0, 255), -1)

    # Players
    red_pts, blue_pts, red_mask, blue_mask = detect_players(hsv, pitch_region)
    for (x, y) in red_pts:  cv2.circle(draw, (x, y), 6, (0, 0, 255), -1)   # Liverpool
    for (x, y) in blue_pts: cv2.circle(draw, (x, y), 6, (255, 0, 0), -1)   # Chelsea

    # Possession (simple nearest within gate)
    owner = None
    possession_txt = "Possession: NONE"
    if bx is not None:
        max_d = 0.10 * diag
        nearest, dr = nearest_player((bx, by), red_pts + blue_pts, max_d)
        owner = nearest
        if owner is not None:
            possession_txt = "Possession: Liverpool" if owner in red_pts else "Possession: Chelsea"
            last_possession_str = possession_txt
        else:
            possession_txt = last_possession_str

    # Count possession per frame
    if possession_txt.endswith("Liverpool"):
        red_frames += 1
    elif possession_txt.endswith("Chelsea"):
        blue_frames += 1
    else:
        none_frames += 1
    frames_counted += 1

    # Draw link to owner
    if owner is not None and bx is not None:
        cv2.line(draw, (bx, by), owner, (50, 220, 50), 2)
        cv2.circle(draw, owner, 10, (50, 220, 50), 2)

    # Overlays
    cv2.putText(draw, possession_txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

    # Show
    cv2.imshow("ball mask", ball_mask)
    cv2.imshow("Possession (video)", draw)

    if writer is not None:
        writer.write(draw)

    key = cv2.waitKey(int(1000//fps)) & 0xFF
    if key == 27 or key == ord('q'):
        break

    frame_idx += 1

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()

print("Video processing done")
print(f"Wrote to: {OUT_PATH}" if WRITE_OUT else "No file written.")

# --- Final possession result ---
total = max(1, red_frames + blue_frames + none_frames)
none_pct = 100.0 * none_frames / total
red_pct  = 100.0 * red_frames  / total + none_pct
blue_pct = 100.0 * blue_frames / total


print("\n=== Possession over the whole video ===")
print(f"Liverpool: {red_frames} frames = {red_pct:.1f}%")
print(f"Chelsea:   {blue_frames} frames = {blue_pct:.1f}%")
print()
print(f"In doubt:      {none_frames} frames = {none_pct:.1f}%")
print(f"Total:     {total} frames")

save_possession_summary(red_pct, blue_pct)
