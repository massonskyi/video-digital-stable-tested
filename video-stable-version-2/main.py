# import required libraries
import numpy as np
from vidgear.gears import VideoGear
import cv2


# define suitable tweak parameters for your stream.
options = {
    "CAP_PROP_FRAME_WIDTH": 320, # resolution 320x240
    "CAP_PROP_FRAME_HEIGHT": 240,
    "CAP_PROP_FPS": 60, # framerate 60fps
}

# To open live video stream on webcam at first index(i.e. 0) 
# device and apply source tweak parameters
stream = VideoGear(source=0, stabilize=True,logging=False, **options).start()

# loop over
while True:

    # read frames from stream
    frame_stab, frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # concatenate both frames
    output_frame = np.concatenate((frame, frame_stab), axis=1)

# put text over concatenated frame
    cv2.putText(
        output_frame,
        "Before",
        (10, output_frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        output_frame,
        "After",
        (output_frame.shape[1] // 2 + 10, output_frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    # Show output window
    cv2.imshow("Stabilized Frame", output_frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()