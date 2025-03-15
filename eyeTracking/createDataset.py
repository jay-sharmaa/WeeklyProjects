import cv2

# Initialize webcam
cam = cv2.VideoCapture('./eyeTracking/fronteyes.mp4')
if not cam.isOpened():
    print("Error: Could not open camera")
    exit()

currentFrame = 0
selected_roi = None

def select_roi(frame):
    """ Function to select region of interest (ROI) """
    roi = cv2.selectROI("Select Region", frame, False)
    cv2.destroyWindow("Select Region")
    if roi == (0, 0, 0, 0):
        print("No ROI selected. Exiting.")
        exit()
    return roi

# Try getting an initial frame for ROI selection
for _ in range(10):  # Retry mechanism in case of delay
    ret, init_frame = cam.read()
    if ret:
        break
if not ret:
    print("Failed to capture initial frame. Exiting.")
    exit()

selected_roi = select_roi(init_frame)
print(f"Selected region: {selected_roi}")

# Main loop to capture frames
while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Process ROI
    if selected_roi is not None:
        x, y, w, h = selected_roi

        # Draw ROI on frame
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Frame with Selection", display_frame)

        # Crop and save ROI
        roi_frame = frame[y:y + h, x:x + w]
        if roi_frame.size != 0 and currentFrame%30 == 0:
            roi_path = f'./eyeTracking/data/front{currentFrame}.jpg'
            cv2.imwrite(roi_path, roi_frame)
            print(f"Saved ROI frame to {roi_path}")

    currentFrame += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
print(f"Total frames captured: {currentFrame}")
