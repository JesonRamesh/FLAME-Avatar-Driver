import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image

# CONFIGS
MODEL_PATH = 'face_landmarker.task'
VIDEO_PATH = 'example.mov'
USE_WEBCAM = False

def run_face_processor():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True, 
        running_mode=vision.RunningMode.VIDEO
    )

    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0 if USE_WEBCAM else VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp = int(frame_count * (1000 / fps))
        frame_count += 1

        detection_result = detector.detect_for_video(mp_image, timestamp)

        if detection_result.face_blendshapes and frame_count % 30 == 0:
            print(f"\n=== FRAME {frame_count} - Active Blendshapes (score > 0.1) ===")
            scores = detection_result.face_blendshapes[0]
            
            # Show all blendshapes with significant values
            active_shapes = [(b.category_name, b.score) for b in scores if b.score > 0.1]
            active_shapes.sort(key=lambda x: x[1], reverse=True)
            
            for name, score in active_shapes:
                print(f"  {name:25s}: {score:.3f}")

        cv2.imshow('MediaPipe Face Landmarker', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_processor()