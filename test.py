import argparse
import cv2
from glob import glob

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerKCF_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}
parser = argparse.ArgumentParser()
parser.add_argument('-t', "--tracker", type=str, default='csrt')
parser.add_argument('-v', '--video_path', type=str, default='./images/')
parser.add_argument('--show', default=True, action='store_true')
args = parser.parse_args()


def main():
    tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()
    frames_path = glob(args.video_path + '/*.jpg')
    frames_path = sorted(frames_path)
    # bbox = [290, 135, 338, 304]
    print(frames_path[0])
    first_frame = cv2.imread(frames_path[0])
    if args.show:
        cv2.imshow("Frame", first_frame)
        # key = cv2.waitKey(1) & 0xFF
        bbox = cv2.selectROI("Frame", first_frame, fromCenter=False, showCrosshair=True)
    # else:
    #     bbox = [290, 135, 338, 304]  # only for this video
    print("first box:", bbox)
    success = tracker.init(first_frame, bbox)
    if not success:
        raise "tracker init error"
    for frame_path in frames_path[1:]:
        frame = cv2.imread(frame_path)
        h, w = frame.shape[0], frame.shape[1]
        success, bbox = tracker.update(frame)
        # print(bbox)
        if success:
            x, y, w, h = bbox
            print("x,y,w,h:", x, y, w, h)
            if args.show:
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        if args.show:
            cv2.putText(frame, "success" if success else "failure", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv2.imshow("Frame", frame)
            cv2.waitKey(25)


if __name__ == "__main__":
    main()