import cv2 as cv
import mediapipe as mp


BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def face_detection_callback(
    result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int
):
    print(result)


def pose_detection_callback(
    result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int
):
    print(result)


def initialise_detectors():
    face_options = FaceDetectorOptions(
        base_options=BaseOptions(
            model_asset_path="models/blaze_face_short_range.tflite"
        ),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=face_detection_callback,
    )
    face_detector = FaceDetector.create_from_options(face_options)

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/pose_landmarker_lite.task"),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=pose_detection_callback,
    )
    pose_detector = PoseLandmarker.create_from_options(pose_options)
    return face_detector, pose_detector


def main():
    face_detector, pose_detector = initialise_detectors()
    count = 0
    cap = cv.VideoCapture(0)

    if not (cap.isOpened()):
        print("Cannot open camera")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        count += 1

        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        cv.imshow("frame", frame)
        if cv.waitKey(1) == ord("q"):
            break

        face_detector.detect_async(image, count)
        pose_detector.detect_async(image, count)

    face_detector.close()
    pose_detector.close()
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
