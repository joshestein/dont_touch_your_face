import cv2 as cv
import mediapipe as mp


BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


FINGER_INDEXES = [
    15,  # left wrist
    16,  # right wrist
    17,  # left pinky
    18,  # right pinky
    19,  # left index
    20,  # right index
    21,  # left thumb
    22,  # right thumb
]

FACE_INDEXES = [
    0,  # nose
    2,  # left eye
    5,  # right eye
    7,  # left ear
    8,  # right ear
    9,  # mouth (left)
    10,  # mouth (right)
]

DISTANCE_THRESHOLD = 0.65


def calculate_distance(first, second):
    return abs(first.x - second.x) + abs(first.y - second.y) + abs(first.z - second.z)


def pose_detection_callback(
    result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    if not result.pose_landmarks:
        return

    normalized_landmarks = result.pose_landmarks[0]

    for finger_index in FINGER_INDEXES:
        if normalized_landmarks[finger_index].presence < 0.9:
            continue

        for face_index in FACE_INDEXES:
            if normalized_landmarks[face_index].presence < 0.9:
                continue

            distance = calculate_distance(
                normalized_landmarks[finger_index], normalized_landmarks[face_index]
            )

            if distance < DISTANCE_THRESHOLD:
                print(distance)


def initialise_detector():
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/pose_landmarker_lite.task"),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=pose_detection_callback,
    )
    pose_detector = PoseLandmarker.create_from_options(pose_options)
    return pose_detector


def main():
    pose_detector = initialise_detector()
    count = 0
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
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

        pose_detector.detect_async(image, count)

    pose_detector.close()
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
