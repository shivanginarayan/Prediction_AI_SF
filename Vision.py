import cv2
import numpy as np

print("OpenCV Test")
# Load picture
image = cv2.imread(r"C:\Users\mathi\OneDrive\Uni\SanFran\Undervisning\AI\VSCodeVirtual\Soccer_AI\Soccerfield.png")

if image is None:
    print("Couldn't load file, check path")
    exit()


# Original picture
resized_image = cv2.resize(image, (1000, 1000))
cv2.imshow("Original", resized_image)
cv2.waitKey(0)

# Convert to grayscale
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Graatoner", gray)

################## FIND RED CIRCLES ##################

# Covert to HSV 
hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

# Hue Saturation Value, for red
lower_red1 = np.array([0, 100, 100])      # darker red
upper_red1 = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

# Clean for noice
kernel = np.ones((5, 5), np.uint8)
mask_clean = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel, iterations=1)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

# Find counters - areas with same color
contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

circles = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 50:
        continue  # removes small noise

    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue

    # Locate circular shapes, 1 = perfect circle
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if circularity > 0.7:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        x, y, radius = int(x), int(y), int(radius)
        circles.append((x, y, radius))
        # Draw circle on image
        cv2.circle(resized_image, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(resized_image, (x, y), 2, (0, 255, 0), -1)

# View picutre with and without mask
cv2.imshow("Red mask (clean)", mask_clean)
cv2.imshow("Detected red circles", resized_image)


# Print number of circles found
print(f"Number of red cirkeles: {len(circles)}")

################## FIND RED CIRCLES - END ##################

# wait for key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()