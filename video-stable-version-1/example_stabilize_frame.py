import cv2
import numpy as np
from src import VidStab, layer_overlay, download_ostrich_video
# Init stabilizer and video reader
stabilizer = VidStab()
vidcap = cv2.VideoCapture(0)

while True:
    grabbed_frame, frame = vidcap.read()

        # Do frame pre-processing
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to stabilizer even if frame is None
    stabilized_frame = stabilizer.stabilize_frame(input_frame=frame)

    # If stabilized_frame is None then there are no frames left to process
    if stabilized_frame is None:
        break
    
    combined_frame = np.hstack((frame, stabilized_frame))
    # Display stabilized output
    cv2.imshow('Stabilized Frame', combined_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

vidcap.release()
cv2.destroyAllWindows()
