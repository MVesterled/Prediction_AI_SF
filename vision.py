import cv2
import numpy as np

############# Player detection and attack inference #############


############# Load and preprocess image #############
# Load picture
image = cv2.imread(r"C:\Users\mathi\OneDrive\Uni\SanFran\Undervisning\AI\VSCodeVirtual\Soccer_AI\Soccerfield4.png")
if image is None:
    print("Couldn't load file, check path")
    raise SystemExit

# Normalize size for consistent processing
resized_image = cv2.resize(image, (1000, 1000))
cv2.imshow("Original", resized_image)
cv2.waitKey(0)

# Convert to HSV once
hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

############# Load and preprocess image - END #############

############# Detect players by colormask and finding circles #############
# --- Color masks ---
# Red uses two hue intervals because red wraps around 0/180 in HSV
lower_red1 = np.array([0,   100, 100])
upper_red1 = np.array([10,  255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])
red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))

# Blue typical hue range in OpenCV's 0-179 scale
lower_blue = np.array([100, 120, 80])
upper_blue = np.array([130, 255, 255])
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Morphological cleaning to remove small noise and fill tiny gaps
kernel = np.ones((5, 5), np.uint8)
def clean_mask(mask):
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return m

red_mask_clean = clean_mask(red_mask)
blue_mask_clean = clean_mask(blue_mask)

# Common detector: find near-circular contours and draw them
def find_circles_from_mask(mask_clean, draw_color_bgr, img_draw, min_area=50, circ_thresh=0.7):
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)  # 1.0 = perfect circle
        if circularity > circ_thresh:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            x, y, radius = int(x), int(y), int(radius)
            found.append((x, y, radius))
            cv2.circle(img_draw, (x, y), radius, draw_color_bgr, 2)
            cv2.circle(img_draw, (x, y), 2, draw_color_bgr, -1)
    return found

# Detect red and blue players
red_circles  = find_circles_from_mask(red_mask_clean,  (0, 0, 255), resized_image)
blue_circles = find_circles_from_mask(blue_mask_clean, (255, 0, 0), resized_image)

# Show masks for debugging
cv2.imshow("Red mask (clean)", red_mask_clean)
cv2.imshow("Blue mask (clean)", blue_mask_clean)

############# Detect players by colormask and finding circles - END #############

############# Count players on each half #############
# Count players in top and bottom halves
h, w = resized_image.shape[:2]
mid_y = h // 2
cv2.line(resized_image, (0, mid_y), (w, mid_y), (255, 255, 255), 2)  # draw midline

red_top    = sum(1 for (x, y, r) in red_circles  if y <  mid_y)
red_bottom = sum(1 for (x, y, r) in red_circles  if y >= mid_y)
blue_top   = sum(1 for (x, y, r) in blue_circles if y <  mid_y)
blue_bottom= sum(1 for (x, y, r) in blue_circles if y >= mid_y)

# Console output
print(f"Number of red players: {len(red_circles)}")
print(f"Number of blue players: {len(blue_circles)}")
print(f"Number of red players on top half: {red_top}")
print(f"Number of blue players on top half: {blue_top}")
print(f"Number of red players on bottom half: {red_bottom}")
print(f"Number of blue players on bottom half: {blue_bottom}")

# On-image overlays
cv2.putText(resized_image, f"Red total: {len(red_circles)}  Top: {red_top}  Bottom: {red_bottom}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(resized_image, f"Blue total: {len(blue_circles)} Top: {blue_top} Bottom: {blue_bottom}",
            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

############# Count players on each half - END #############


################# Is attacking?  #################
# --- Attack inference (simple pressure rule) ---
# Red is home on the TOP half. "Attackers" are players on the opponent's half.
red_attackers  = red_bottom   # red on away half (bottom)
blue_attackers = blue_top     # blue on away half (top)

# Margin scales with team size (20% of larger team, at least 1)
margin = max(1, int(0.2 * max(len(red_circles), len(blue_circles))))

pressure = red_attackers - blue_attackers  # >0 favors RED, <0 favors BLUE

if pressure >= margin:
    attack_team = "RED (likely attacking)"
elif pressure <= -margin:
    attack_team = "BLUE (likely attacking)"
else:
    attack_team = "UNSURE (balanced)"

# On-image overlay
cv2.putText(resized_image, f"Attack: {attack_team}",
            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 220, 50), 2, cv2.LINE_AA)

print(f"Attack:", attack_team)

################# Is attacking - END  #################

# Final visualization
cv2.imshow("Detected players (red + blue)", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()





