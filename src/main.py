import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from translator import FlameTranslator 
from visualizer import Visualizer

# --- Configuration ---
MODEL_PATH = 'face_landmarker.task'
FLAME_PATH = 'model/generic_model.pkl'
MAPPINGS_PATH = './mappings'  # Will fallback to manual if not found
VIDEO_PATH = 'examples/example2.mov'

def get_head_rotation(landmarks):
    """Extract head rotation from MediaPipe landmarks."""
    nose = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
    left_eye = np.array([landmarks[33].x, landmarks[33].y, landmarks[33].z])
    right_eye = np.array([landmarks[263].x, landmarks[263].y, landmarks[263].z])

    eye_center = (left_eye + right_eye) / 2
    forward = nose - eye_center

    yaw = np.arctan2(forward[0], forward[2])
    pitch = np.arctan2(forward[1], forward[2])

    return yaw, pitch

def main():
    # Initialize translator (will auto-detect if mappings are available)
    print("--- Initializing FLAME Translator ---")
    translator = FlameTranslator(FLAME_PATH, mappings_path=MAPPINGS_PATH)
    visualizer = Visualizer(translator.model['f'])
    
    # Setup MediaPipe
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    print("\nStarting Real-Time FLAME Avatar Driver")
    print(f"FPS: {fps}")
    print("Press 'q' to quit\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        # Convert frame to MediaPipe format
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, 
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp = int(1000 * frame_count / fps)
        frame_count += 1

        # Detect face
        result = detector.detect_for_video(rgb_frame, timestamp)

        if result.face_blendshapes:
            mp_scores = result.face_blendshapes[0]
            
            # Translate blendshapes to FLAME coefficients
            translation_result = translator.translate(mp_scores)
            
            # Handle both return formats (with/without mappings)
            if len(translation_result) == 3:
                flame_expr, jaw_pose, eye_pose = translation_result
            else:
                flame_expr = translation_result
                jaw_pose = None
                eye_pose = None
            
            # Apply expression and pose to deform the mesh
            deformed_vertices = translator.deform_mesh(flame_expr, jaw_pose)

            # Get head rotation from landmarks
            landmarks = result.face_landmarks[0]
            raw_yaw, raw_pitch = get_head_rotation(landmarks)

            # Invert yaw for FLAME coordinate system
            yaw = -raw_yaw
            pitch = raw_pitch
            
            # Rotation Matrices
            R_pitch = np.array([
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)]
            ])
            R_yaw = np.array([
                [np.cos(yaw), 0, np.sin(yaw)],
                [0, 1, 0],
                [-np.sin(yaw), 0, np.cos(yaw)]
            ])

            # Base orientation corrections for FLAME model
            R_z = np.array([
                [np.cos(np.radians(180)), -np.sin(np.radians(180)), 0],
                [np.sin(np.radians(180)), np.cos(np.radians(180)), 0],
                [0, 0, 1]
            ])

            R_x = np.array([
                [1, 0, 0],
                [0, np.cos(np.radians(-35)), -np.sin(np.radians(-35))],
                [0, np.sin(np.radians(-35)), np.cos(np.radians(-35))]
            ])

            
            # Apply the transformations
            deformed_vertices = deformed_vertices @ R_z.T @ R_x.T
            deformed_vertices = deformed_vertices @ R_yaw.T @ R_pitch.T
            
            # Center the mesh
            deformed_vertices -= np.mean(deformed_vertices, axis=0)
            
            # Update the avatar visualization
            visualizer.update_mesh(deformed_vertices)
            
            # Debug
            if frame_count % 30 == 0: 
                print(f"Frame {frame_count}:")
                print(f"Expression range: [{flame_expr.min():.3f}, {flame_expr.max():.3f}]")
                if jaw_pose is not None:
                    print(f"Jaw pose: [{jaw_pose[0]:.3f}, {jaw_pose[1]:.3f}, {jaw_pose[2]:.3f}]")
                if eye_pose is not None:
                    print(f"Eye pose: [{eye_pose[0]:.3f}, {eye_pose[1]:.3f}, ...]")
                print(f"Head rotation: yaw={yaw:.3f}, pitch={pitch:.3f}")
                
                # Show active expressions
                active_expr = np.where(np.abs(flame_expr) > 0.1)[0]
                if len(active_expr) > 0:
                    print(f"Active expressions: {active_expr[:5]}...")

        # Show the video feed
        cv2.imshow('Avatar Driver Pipeline', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n=== Session ended ===")

if __name__ == "__main__":
    main()