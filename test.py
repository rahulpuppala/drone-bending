import cv2
import mediapipe as mp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MediaPipe drawing utilities and the Holistic model
logger.info("Initializing MediaPipe components...")
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Start capturing video from the webcam. 0 usually means the default webcam.
# On Windows, use DirectShow backend (CAP_DSHOW) to avoid hanging
logger.info("Attempting to open webcam with DirectShow backend...")
cap = None

# Try different camera indices with DirectShow backend
for camera_index in [0, 1, 2]:
    logger.info(f"Trying camera index {camera_index}...")
    test_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if test_cap.isOpened():
        # Test if we can actually read a frame
        ret, test_frame = test_cap.read()
        if ret:
            logger.info(f"Successfully opened camera at index {camera_index}")
            cap = test_cap
            break
        else:
            logger.warning(f"Camera {camera_index} opened but couldn't read frame")
            test_cap.release()
    else:
        logger.warning(f"Could not open camera at index {camera_index}")

if cap is None or not cap.isOpened():
    logger.error("ERROR: Could not open any webcam. Please check if:")
    logger.error("  1. Your webcam is connected")
    logger.error("  2. No other application is using the webcam")
    logger.error("  3. You have granted camera permissions")
    logger.error("  4. Try closing other apps that might use the camera (Teams, Zoom, etc.)")
    exit(1)

# Get webcam properties
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
logger.info(f"Webcam opened successfully: {width}x{height} @ {fps} FPS")

# Initialize the holistic model
logger.info("Initializing Holistic model...")
frame_count = 0

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    
    logger.info("Holistic model initialized. Starting video capture loop...")
    logger.info("Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        
        if not ret:
            logger.warning(f"Failed to read frame {frame_count}")
            continue

        if frame_count == 1:
            logger.info(f"First frame captured successfully: shape={frame.shape}")

        # Convert the BGR image to RGB and process with the holistic model
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)

        # Log detection results every 30 frames
        if frame_count % 30 == 0:
            logger.info(f"Frame {frame_count}: "
                       f"Pose={'✓' if results.pose_landmarks else '✗'} "
                       f"Left Hand={'✓' if results.left_hand_landmarks else '✗'} "
                       f"Right Hand={'✓' if results.right_hand_landmarks else '✗'} "
                       f"Face={'✓' if results.face_landmarks else '✗'}")

        # Convert back to BGR and draw landmarks
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw pose, hand, and face landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.face_landmarks:
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

        # Display the processed frame
        cv2.imshow('MediaPipe Holistic Detection', image)

        # Break the loop when 'q' is pressed
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            logger.info("User pressed 'q' - exiting...")
            break

logger.info(f"Total frames processed: {frame_count}")
logger.info("Releasing webcam and closing windows...")

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

logger.info("Program finished successfully")
