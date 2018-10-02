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
    parser.add_argument("-v", "--video", type=str,help="path to input video file")
    args = vars(parser.parse_args())

    vs = VideoStream(args["video"]).start()
    time.sleep(2.0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # max number of hands we want to detect/track
    num_hands_detect = 2
    score_thresh = 0.5

    while True:
        frame_count += 1
        frame = vs.read()
        if frame is None:
            break
        framez = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        print(h,":",w)

        # actual detection
        boxes, scores = detector_utils.detect_objects(framez, detection_graph, sess)

        # draw bounding boxes
        detector_utils.draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, w, h,
                                         frame)
        namez = '{0}.jpg'.format(frame_count)
        
        name = '/content/videos/'+namez
        cv2.imwrite(name, frame)
    cv2.destroyAllWindows()
    vs.stop()
