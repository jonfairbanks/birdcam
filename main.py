import argparse
import cv2
from datetime import datetime, timedelta
from notifications import post_message_to_slack
from notifications import post_file_to_slack
from object_detector import ObjectDetector
from object_detector import ObjectDetectorOptions
from image_classifier import ImageClassifier
from image_classifier import ImageClassifierOptions
import utils
import os
import time
import threading
from flask import Response, Flask, send_from_directory
import imageio

__version__ = "1.0.0"

# Arugment handling
parser = argparse.ArgumentParser(description="Raspberry Pi based camera monitor for detecting and classifying birds with notification support", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--debug", help="Increase output verbosity", action="store_true")
parser.add_argument("-v", "--version", help="Current Birdcam version.", action="store_true")
parser.add_argument("--slack-token", help="Slack bot token to be used for notifications")
parser.add_argument("--slack-channel", help="Slack channel to be used for notifications")
parser.add_argument("--detection-delay", help="Interval in seconds between detections", required=False, type=int, default=60)
parser.add_argument("--disable-detection", help="Disable object detection", action="store_true")
parser.add_argument("--port", help="Web Port", default=8000, type=int)
parser.add_argument("--save", help="Archive detected objects", action="store_true")

parser.add_argument('--model', help='Path of the object detection model.', required=False, default='efficientdet_lite0.tflite')
parser.add_argument('--classification-model', help='Path of the image classification model.', required=False, default='lite-model_aiy_vision_classifier_birds_V1_3.tflite')
parser.add_argument('--cameraId', help='Id of camera.', required=False, type=int, default=0)
parser.add_argument('--frameWidth', help='Width of frame to capture from camera.', required=False, type=int, default=1280)
parser.add_argument('--frameHeight', help='Height of frame to capture from camera.', required=False, type=int, default=720)
parser.add_argument('--numThreads', help='Number of CPU threads to run the model.', required=False, type=int, default=1)
parser.add_argument('--flipFrame', help='Flip orientation of camera', required=False, type=int, default=0)

# Global variable definitions
args = parser.parse_args() 
video_frames = [None] * 10 # initialize array of empty frames

last_detection_time = datetime.now()

# Open and configure camera source
video_capture = cv2.VideoCapture(args.cameraId)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.frameWidth)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frameHeight)
video_capture.set(cv2.CAP_PROP_FPS, 15)
#video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, .75) # Disable auto exposure
#video_capture.set(cv2.CAP_PROP_EXPOSURE, 30) # Set exposure

# Define flask
app = Flask(__name__)

# Read frames from camera source and store globally
def captureFrames():
    global video_frames
    capture_thread_lock = threading.Lock()
    counter = 0
    fps = 0
    start_time = time.time()

    while True and video_capture.isOpened():
        counter += 1
        with capture_thread_lock:

            # Read Frames
            return_key, frame = video_capture.read()
            if not return_key:
                break

            # Flip Frame
            frame = cv2.flip(frame, args.flipFrame)

            if args.debug:
                # Calculate the FPS
                fps_avg_frame_count = 10
                if counter % fps_avg_frame_count == 0:
                    end_time = time.time()
                    fps = fps_avg_frame_count / (end_time - start_time)
                    start_time = time.time()

                # Show the FPS
                fps_text = 'FPS = {:.1f}'.format(fps)
                text_location = (24, 20)
                cv2.putText(frame, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    1, (0, 0, 255), 1)

            # Save frame to global variable
            video_frames.insert(0,frame.copy())

            # Truncate frames (insert + truncate probably needs to be improved)
            video_frames = video_frames[0:90]

    # release video stream        
    video_capture.release()


# Detect objects in frame and notify
def detectObject():
    global video_frames, last_detection_time

    # Initialize the object detection model
    options = ObjectDetectorOptions(
        num_threads=args.numThreads,
        score_threshold=0.6,
        max_results=10,
        label_allow_list=["bird","person"])
    detector = ObjectDetector(model_path=args.model, options=options)

    while True:
        global video_frames
        if video_frames[0] is None:
            continue

        # Run object detection estimation using the model.   
        detections = detector.detect(video_frames[0])

        # Draw detection bounding boxes
        if args.debug:
            video_frames[0] = utils.visualize(video_frames[0], detections)

        # Check if detection should run
        seconds_since_notified = (datetime.now() - last_detection_time).total_seconds()
        if (seconds_since_notified > args.detection_delay):

            if len(detections) > 0:
                detection_frame = video_frames[0]
                recording = True
                start_recording_time = datetime.now()
                timestamp = format(datetime.now().strftime("%m%d%Y%H%M%S"))

                while recording:
                    seconds_recording = (datetime.now() - start_recording_time).total_seconds()
                    if (seconds_recording > 4):
                        recording = False

                # Reset last detection timestamp
                last_detection_time = datetime.now()



                # Initialize classification model
                options = ImageClassifierOptions(
                    num_threads=args.numThreads,
                    max_results=10)
                classifier = ImageClassifier(args.classification_model, options)

                # Crop detection frame for classification
                detection_frame = utils.cropDetection(detection_frame, detections[0])
                # Resize to 244px
                detection_frame = utils.resize(detection_frame)

                # Classify detected_frame
                categories = classifier.classify(detection_frame)

                # Show classification results on the image
                for idx, category in enumerate(categories):
                    class_name = category.label
                    
                    score = round(category.score, 2)
                    result_text = class_name + ' (' + str(score) + ')'
                    print(result_text)
                    text_location = (24, (idx + 2) * 20)
                    cv2.putText(detection_frame, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,1, (0, 0, 255), 1)

                
                
                # Iterate detections
                for detection in detections:

                    # Capture label
                    detection_label = detection.categories[0].label
                    
                    # Save detection frame to jpg file
                    cv2.imwrite('motion-%s.jpg' % timestamp, detection_frame)

                    # Write frames to video file
                    out = cv2.VideoWriter('motion-%s.mp4' % timestamp, cv2.VideoWriter_fourcc('H','2','6','4'), 15, (1280,720))
                    for frame in reversed(video_frames):
                        out.write(frame)
                    out.release()

                    # Send notification
                    if args.slack_token and args.slack_channel:
                        return_key, encoded_image = cv2.imencode(".jpg", video_frames[0])
                        if not return_key:
                            continue
                        try:
                            video_file = open('motion-%s.mp4' % timestamp, "rb")
                            image_file = open('motion-%s.jpg' % timestamp, "rb")

                            post_file_to_slack(
                                '%s detected' % detection_label,
                                args,
                                'motion-%s.jpg' % timestamp,
                                image_file)
                            print(f"Successfully posted image to Slack")

                            post_file_to_slack(
                                '%s detected' % detection_label,
                                args,
                                'motion-%s.mp4' % timestamp,
                                video_file)
                            print(f"Successfully posted video to Slack")

                        except Exception as e:
                            print(f"Failed to post a notification message: {e}" )
                            continue
                    
                    # Remove capture files if not configured to save
                    if not args.save:
                        os.remove('motion-%s.mp4' % timestamp)
                        os.remove('motion-%s.jpg' % timestamp)


def encodeFrames():
    encode_thread_lock = threading.Lock()
    while True:
        with encode_thread_lock:
            global video_frames
            if video_frames[0] is None:
                continue
            return_key, encoded_image = cv2.imencode(".jpg", video_frames[0])
            if not return_key:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n') # Output image as a byte array


@app.route('/')  
def index():  
    return send_from_directory(app.static_folder, 'index.html')

#@app.route('/<path:filename>')  
#def send_file(filename):  
#    return send_from_directory(app.static_folder, filename)

@app.route("/live")
def streamFrames():
    return Response(encodeFrames(), mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    if args.version:
        print(f"Version: {__version__}")

    if args.slack_token:
        print("\n** Slack Notifications: ENABLED **")

    if args.debug:
        print("\n** Debug Mode: ENABLED **")

    if args.disable_detection:
        print("\n** Object Detection: DISABLED **")

    try:
        capture_thread = threading.Thread(target=captureFrames)
        capture_thread.daemon = True
        capture_thread.start()

        if not args.disable_detection:
            detection_thread = threading.Thread(target=detectObject)
            detection_thread.daemon = True
            detection_thread.start()

        app.run("0.0.0.0", port=args.port)
        
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
