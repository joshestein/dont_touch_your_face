import cv2 as cv
import mediapipe as mp


BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode


def detection_callback(
    result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int
):
    print(result)


def main():
    options = FaceDetectorOptions(
        base_options=BaseOptions(
            model_asset_path="models/blaze_face_short_range.tflite"
        ),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=detection_callback,
    )

    count = 0
    cap = cv.VideoCapture(0)

    if not (cap.isOpened()):
        print("Cannot open camera")
        exit()

    with FaceDetector.create_from_options(options) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            count += 1

            if not ret:
                print("Can't receive fram (stream end?). Exiting...")
                break

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            cv.imshow("frame", frame)
            if cv.waitKey(1) == ord("q"):
                break

            detector.detect_async(image, count)

        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
