from typing import List, Optional
import torch
from torch.nn import DataParallel
import os
import numpy as np
import cv2
import dlib
import util.gaze
from imutils import face_utils
import math

from models.eyenet import EyeNet
from util.eye_prediction import EyePrediction
from util.eye_sample import EyeSample

# Enable cuDNN if available
torch.backends.cudnn.enabled = True

# Select device: GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set up webcam
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
webcam.set(cv2.CAP_PROP_FPS, 60)

dirname = os.path.dirname(__file__)
face_cascade = cv2.CascadeClassifier(
    os.path.join(dirname, "lbpcascade_frontalface_improved.xml")
)
landmarks_detector = dlib.shape_predictor(
    os.path.join(dirname, "shape_predictor_5_face_landmarks.dat")
)

# Load EyeNet model
checkpoint = torch.load("checkpoint.pt", map_location=device)
nstack = checkpoint["nstack"]
nfeatures = checkpoint["nfeatures"]
nlandmarks = checkpoint["nlandmarks"]
eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
eyenet.load_state_dict(checkpoint["model_state_dict"])

# Get frame width and height
width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Frame Width:", width, "Frame Height:", height)

# Define the zigzag path for the ball:
# 1. Top-left to top-right
# 2. Top-right to bottom-left
# 3. Bottom-left to bottom-right
points_path = [
    (50, 50),  # Start top-left
    (width - 50, 50),  # Move horizontally to top-right
    (50, height - 50),  # Move diagonally to bottom-left
    (width - 50, height - 50),  # Move diagonally to bottom-right
]

# We will move the ball along these segments until finished.
# Once we reach the last point, we stop and save results.

records = []  # (x_target, y_target, x_pred, y_pred)


def main():
    current_face = None
    landmarks = None
    alpha = 0.95  # smoothing factor for face/landmarks
    left_eye = None
    right_eye = None

    segment_index = 0
    current_pos = np.array(points_path[0], dtype=float)
    next_point = np.array(points_path[1], dtype=float)
    speed = 20.0

    while True:
        ret, frame_bgr = webcam.read()
        if not ret:
            break

        orig_frame = frame_bgr.copy()
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray)

        # Smooth face bounding box
        if len(faces) > 0:
            next_face = faces[0]
            if current_face is not None:
                current_face = alpha * current_face + (1 - alpha) * next_face
            else:
                current_face = next_face

        # Detect and smooth landmarks
        if current_face is not None:
            next_landmarks = detect_landmarks(current_face, gray)
            if landmarks is not None:
                landmarks = alpha * landmarks + (1 - alpha) * next_landmarks
            else:
                landmarks = next_landmarks

        # Move the ball towards the next point in the path
        direction = next_point - current_pos
        dist = np.linalg.norm(direction)
        if dist < speed:
            current_pos = next_point
            segment_index += 1
            if segment_index < len(points_path) - 1:
                next_point = np.array(points_path[segment_index + 1], dtype=float)
            else:
                pass
        else:
            direction = direction / dist
            current_pos = current_pos + direction * speed

        # Draw the yellow ball
        target_point = (int(current_pos[0]), int(current_pos[1]))
        cv2.circle(orig_frame, target_point, 20, (0, 255, 255), -1)

        predicted_gaze_point = None

        # Process landmarks and run EyeNet
        if landmarks is not None:
            eye_samples = segment_eyes(gray, landmarks)
            eye_preds = run_eyenet(eye_samples)

            # Separate left and right eyes
            left_eyes = list(filter(lambda x: x.eye_sample.is_left, eye_preds))
            right_eyes = list(filter(lambda x: not x.eye_sample.is_left, eye_preds))

            # Smooth predictions
            if left_eyes:
                left_eye = smooth_eye_landmarks(left_eyes[0], left_eye, smoothing=0.1)
            if right_eyes:
                right_eye = smooth_eye_landmarks(
                    right_eyes[0], right_eye, smoothing=0.1
                )

            predicted_points = []

            # Process each eye prediction
            for ep in [left_eye, right_eye]:
                if ep is None:
                    continue

                # Draw gaze landmarks and compute predictions
                for x, y in ep.landmarks[16:33]:
                    color = (0, 255, 0)
                    if ep.eye_sample.is_left:
                        color = (255, 0, 0)
                    cv2.circle(
                        orig_frame,
                        (int(round(x)), int(round(y))),
                        1,
                        color,
                        -1,
                        lineType=cv2.LINE_AA,
                    )

                gaze = ep.gaze.copy()
                if ep.eye_sample.is_left:
                    gaze[1] = -gaze[1]

                util.gaze.draw_gaze(
                    orig_frame, ep.landmarks[-2], gaze, length=60.0, thickness=2
                )

                eye_center = ep.landmarks[-2]
                length = 60.0
                predicted_gaze_endpoint = (
                    eye_center[0] + gaze[0] * length,
                    eye_center[1] + gaze[1] * length,
                )
                predicted_points.append(predicted_gaze_endpoint)

                cv2.circle(
                    orig_frame,
                    (int(predicted_gaze_endpoint[0]), int(predicted_gaze_endpoint[1])),
                    4,
                    (255, 0, 255) if ep.eye_sample.is_left else (255, 255, 0),
                    -1,
                )

            # Calculate and draw real-time gaze point
            if len(predicted_points) == 2:
                predicted_gaze_point = (
                    (predicted_points[0][0] + predicted_points[1][0]) / 2.0,
                    (predicted_points[0][1] + predicted_points[1][1]) / 2.0,
                )
            elif len(predicted_points) == 1:
                predicted_gaze_point = predicted_points[0]

            if predicted_gaze_point is not None:
                # Draw the real-time gaze point on the screen in red
                cv2.circle(
                    orig_frame,
                    (int(predicted_gaze_point[0]), int(predicted_gaze_point[1])),
                    8,
                    (0, 0, 255),  # Red color
                    -1,
                )

                # Record data
                records.append(
                    (
                        target_point[0],
                        target_point[1],
                        predicted_gaze_point[0],
                        predicted_gaze_point[1],
                    )
                )

        cv2.imshow("Webcam", orig_frame)

        # Check completion of all segments
        if (
            segment_index == len(points_path) - 1
            and np.linalg.norm(next_point - current_pos) < speed
        ):
            break

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()

    # Save results
    save_results(records, "gaze_results.txt")


def detect_landmarks(face, frame):
    (x, y, w, h) = (int(e) for e in face)
    rectangle = dlib.rectangle(x, y, x + w, y + h)
    face_landmarks = landmarks_detector(frame, rectangle)
    return face_utils.shape_to_np(face_landmarks)


def segment_eyes(frame, landmarks, ow=160, oh=96):
    eyes = []
    # corners: (2,3) left eye, (0,1) right eye for given 5-point landmarks
    for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
        x1, y1 = landmarks[corner1, :]
        x2, y2 = landmarks[corner2, :]
        eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
        if eye_width == 0.0:
            return eyes

        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        # Translation
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-cx], [-cy]]
        inv_translate_mat = np.asmatrix(np.eye(3))
        inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

        # Scale
        scale = ow / eye_width
        scale_mat = np.asmatrix(np.eye(3))
        scale_mat[0, 0] = scale_mat[1, 1] = scale
        inv_scale = 1.0 / scale
        inv_scale_mat = np.asmatrix(np.eye(3))
        inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

        estimated_radius = 0.5 * eye_width * scale

        # Center
        center_mat = np.asmatrix(np.eye(3))
        center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
        inv_center_mat = np.asmatrix(np.eye(3))
        inv_center_mat[:2, 2] = -center_mat[:2, 2]

        transform_mat = center_mat * scale_mat * translate_mat
        inv_transform_mat = inv_translate_mat * inv_scale_mat * inv_center_mat

        eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (ow, oh))
        eye_image = cv2.equalizeHist(eye_image)

        # Flip left eye image horizontally
        if is_left:
            eye_image = np.fliplr(eye_image)
            cv2.imshow("left eye image", eye_image)
        else:
            cv2.imshow("right eye image", eye_image)

        eyes.append(
            EyeSample(
                orig_img=frame.copy(),
                img=eye_image,
                transform_inv=inv_transform_mat,
                is_left=is_left,
                estimated_radius=estimated_radius,
            )
        )
    return eyes


def smooth_eye_landmarks(
    eye: EyePrediction,
    prev_eye: Optional[EyePrediction],
    smoothing=0.2,
    gaze_smoothing=0.4,
):
    if prev_eye is None:
        return eye
    return EyePrediction(
        eye_sample=eye.eye_sample,
        landmarks=smoothing * prev_eye.landmarks + (1 - smoothing) * eye.landmarks,
        gaze=gaze_smoothing * prev_eye.gaze + (1 - gaze_smoothing) * eye.gaze,
    )


def run_eyenet(eyes: List[EyeSample], ow=160, oh=96) -> List[EyePrediction]:
    result = []
    for eye in eyes:
        with torch.no_grad():
            x = torch.tensor([eye.img], dtype=torch.float32).to(device)
            _, landmarks, gaze = eyenet.forward(x)
            landmarks = np.asarray(landmarks.cpu().numpy()[0])
            gaze = np.asarray(gaze.cpu().numpy()[0])

            # Rescale landmarks
            landmarks = landmarks * np.array([oh / 48, ow / 80])

            temp = np.zeros((34, 3))
            if eye.is_left:
                temp[:, 0] = ow - landmarks[:, 1]
            else:
                temp[:, 0] = landmarks[:, 1]
            temp[:, 1] = landmarks[:, 0]
            temp[:, 2] = 1.0
            landmarks = temp
            landmarks = np.asarray(np.matmul(landmarks, eye.transform_inv.T))[:, :2]
            result.append(EyePrediction(eye_sample=eye, landmarks=landmarks, gaze=gaze))
    return result


def save_results(records, filename):
    if not records:
        print("No records to save.")
        return

    # Compute L2 distances
    distances = []
    for x_t, y_t, x_p, y_p in records:
        dist = math.sqrt((x_p - x_t) ** 2 + (y_p - y_t) ** 2)
        distances.append(dist)

    avg_dist = sum(distances) / len(distances) if distances else 0.0
    print(f"Average L2 Distance: {avg_dist:.2f}")

    # Save to a txt file
    with open(filename, "w") as f:
        f.write("x_target y_target x_pred y_pred distance\n")
        for (x_t, y_t, x_p, y_p), d in zip(records, distances):
            f.write(f"{x_t} {y_t} {x_p} {y_p} {d}\n")
        f.write(f"Average L2 Distance: {avg_dist}\n")

    print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()
