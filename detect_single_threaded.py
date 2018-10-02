from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
from imutils.video import VideoStream
import imutils
import time

detection_graph, sess = detector_utils.load_inference_graph()
out = cv2.VideoWriter('output.mp4', -1, 20.0, (640, 480))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float,
                        default=0.5, help='Score threshold for displaying bounding boxes')
    parser.add_argument('-fps', '--fps', dest='fps', type=int,
                        default=20, help='Show FPS on detection/display visualization')
    parser.add_argument("-o", "--output", required=True, help="path to output video file")
    parser.add_argument("-c", "--codec", type=str, default="MJPG", help="codec of output video")
    args = parser.parse_args()

    cap = VideoStream('/content/test.MP4').start()
    time.sleep(2.0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    fourcc = cv2.VideoWriter_fourcc(*args["codec"])
    writer = None
    (h, w) = (None, None)
    zeros = None

    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        frame = vs.read()
        framez = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # actual detection
        boxes, scores = detector_utils.detect_objects(framez, detection_graph, sess)

        # draw bounding boxes
        detector_utils.draw_box_on_image(num_hands_detect, args["score_thresh"], scores, boxes, im_width, im_height,
                                         frame)
        if writer is None:
            # store the image dimensions, initialzie the video writer,
            # and construct the zeros array
            (h, w) = frame.shape[:2]
            writer = cv2.VideoWriter(args["output"], fourcc, args["fps"], (w * 2, h * 2), True)
            zeros = np.zeros((h, w), dtype="uint8")

        # break the image into its RGB components, then construct the
        # RGB representation of each frame individually
        (B, G, R) = cv2.split(frame)
        R = cv2.merge([zeros, zeros, R])
        G = cv2.merge([zeros, G, zeros])
        B = cv2.merge([B, zeros, zeros])

        # construct the final output frame, storing the original frame
        # at the top-left, the red channel in the top-right, the green
        # channel in the bottom-right, and the blue channel in the
        # bottom-left
        output = np.zeros((h * 2, w * 2, 3), dtype="uint8")
        output[0:h, 0:w] = frame
        output[0:h, w:w * 2] = R
        output[h:h * 2, w:w * 2] = G
        output[h:h * 2, 0:w] = B

        # write the output frame to file
        writer.write(output)
        if key == ord("q"):
            break

    # do a bit of cleanup
    print("[INFO] cleaning up...")
    cv2.destroyAllWindows()
    vs.stop()
    writer.release()
