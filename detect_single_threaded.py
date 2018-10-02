from utils import detector_utils as detector_utils
import cv2
import datetime
import argparse
from imutils.video import VideoStream
import imutils
import time

detection_graph, sess = detector_utils.load_inference_graph()
frame_count = 0

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fps', '--fps', dest='fps', type=int,
                        default=20, help='Show FPS on detection/display visualization')
    parser.add_argument("-o", "--output", required=True, help="path to output video file")
    parser.add_argument("-c", "--codec", type=str, default="MJPG", help="codec of output video")
    args = parser.parse_args()

    vs = VideoStream('/content/test.MP4').start()
    time.sleep(2.0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    frame_count += 1
    # max number of hands we want to detect/track
    num_hands_detect = 2
    score_thresh = 0.5

    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        frame = vs.read()
        framez = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        print(h,":",w)

        # actual detection
        boxes, scores = detector_utils.detect_objects(framez, detection_graph, sess)

        # draw bounding boxes
        detector_utils.draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, w, h,
                                         frame)
        name = '{0}.jpg'.format(frame_count)
        name = os.path.join('/content/video', name)
        cv2.imwrite(name, frame)
        if frame is None:
            break
    cv2.destroyAllWindows()
    vs.stop()
