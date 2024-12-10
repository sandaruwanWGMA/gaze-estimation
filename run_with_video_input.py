from typing import List, Optional

import torch
from torch.nn import DataParallel

from models.eyenet import EyeNet
import os
import numpy as np
import cv2
import dlib
import imutils
import util.gaze
from imutils import face_utils

from util.eye_prediction import EyePrediction
from util.eye_sample import EyeSample

torch.backends.cudnn.enabled = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

video_path = "molindu_eye_rotating.mp4"
webcam = cv2.VideoCapture(video_path)

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

checkpoint = torch.load("checkpoint.pt", map_location=device)
nstack = checkpoint["nstack"]
nfeatures = checkpoint["nfeatures"]
nlandmarks = checkpoint["nlandmarks"]
eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
eyenet.load_state_dict(checkpoint["model_state_dict"])


def main():
    frame_count = 0  # Initialize a frame counter
    output_file = open(
        "gaze_estimation_results.txt", "w"
    )  # Open a file to write results

    while True:
        ret, frame_bgr = webcam.read()
        if not ret:
            break  # Stop the loop if no frames are left

        frame_count += 1  # Increment the frame counter
        frame_info = f"Processing frame #{frame_count}\n"
        print(frame_info)
        output_file.write(frame_info)

        orig_frame = frame_bgr.copy()
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        faces_info = f"Detected {len(faces)} faces\n"
        print(faces_info)
        output_file.write(faces_info)

        for x, y, w, h in faces:
            face_info = f"Landmarks for face at ({x}, {y}, {w}, {h}): "
            cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_rect = dlib.rectangle(x, y, x + w, y + h)
            landmarks = landmarks_detector(gray, face_rect)
            landmarks = face_utils.shape_to_np(landmarks)
            face_info += f"{landmarks}\n"
            print(face_info)
            output_file.write(face_info)

            # Draw the detected landmarks on the frame with different colors
            draw_colored_landmarks(landmarks, orig_frame)

            # Segment eyes from the landmarks
            eye_samples = segment_eyes(gray, landmarks)

            for eye in eye_samples:
                eye_type = "Left" if eye.is_left else "Right"
                print(
                    f"{eye_type} eye segmented with estimated radius: {eye.estimated_radius}"
                )

            if not eye_samples:
                continue

            # Run gaze estimation model on the segmented eyes
            eye_predictions = run_eyenet(eye_samples)

            # Variables to store gaze angles for both eyes
            left_gaze = None
            right_gaze = None

            for eye_pred in eye_predictions:
                gaze_info = f"Eye prediction for {'left' if eye_pred.eye_sample.is_left else 'right'} eye: Landmarks {eye_pred.landmarks}, Gaze {eye_pred.gaze}\n"
                print(gaze_info)
                output_file.write(gaze_info)

                # Assign gaze angles to left_gaze or right_gaze
                if eye_pred.eye_sample.is_left:
                    left_gaze = eye_pred.gaze
                else:
                    right_gaze = eye_pred.gaze

                # Draw the predicted eye landmarks in a different color
                prediction_color = (
                    (255, 0, 255) if eye_pred.eye_sample.is_left else (255, 255, 0)
                )
                for px, py in eye_pred.landmarks:
                    cv2.circle(
                        orig_frame,
                        (int(px), int(py)),
                        2,
                        prediction_color,
                        -1,
                        lineType=cv2.LINE_AA,
                    )

                # Draw the gaze direction arrow from pitch/yaw
                eye_center = np.mean(eye_pred.landmarks, axis=0)
                pitch = eye_pred.gaze[0]
                yaw = eye_pred.gaze[1]

                scale_factor = 50  # Adjust this for visibility
                dx = scale_factor * yaw
                dy = scale_factor * pitch
                end_point = (eye_center[0] + dx, eye_center[1] + dy)

                cv2.arrowedLine(
                    orig_frame,
                    (int(eye_center[0]), int(eye_center[1])),
                    (int(end_point[0]), int(end_point[1])),
                    (0, 255, 255),  # color of the arrow (yellow)
                    2,  # thickness
                    cv2.LINE_AA,  # line_type
                    0,  # shift
                    0.3,  # tipLength
                )

            # Once we have both left and right gaze:
            if left_gaze is not None and right_gaze is not None:
                pitch_left, yaw_left = left_gaze
                pitch_right, yaw_right = right_gaze

                direction_left = spherical_to_cartesian(pitch_left, yaw_left)
                direction_right = spherical_to_cartesian(pitch_right, yaw_right)

                dot_product = np.dot(direction_left, direction_right)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                vergence_angle = np.arccos(dot_product)
                print("Vergence angle:", vergence_angle)
                output_file.write(f"Vergence angle: {vergence_angle}\n")

        # Add a blank line after processing each frame for better readability
        print("\n")
        output_file.write("\n")

        cv2.imshow("Processed Frame", orig_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()
    output_file.close()  # Close the file when done


def detect_landmarks(face, frame, scale_x=0, scale_y=0):
    (x, y, w, h) = (int(e) for e in face)
    rectangle = dlib.rectangle(x, y, x + w, y + h)
    face_landmarks = landmarks_detector(frame, rectangle)
    return face_utils.shape_to_np(face_landmarks)


def draw_cascade_face(face, frame):
    (x, y, w, h) = (int(e) for e in face)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


def draw_colored_landmarks(landmarks, frame):
    # Landmark indices:
    # 0,1 = Right eye corners
    # 2,3 = Left eye corners
    # 4 = Nose tip

    # Define colors in BGR
    right_eye_color = (0, 0, 255)  # Red for right eye
    left_eye_color = (255, 0, 0)  # Blue for left eye
    nose_color = (0, 255, 0)  # Green for nose

    # Draw right eye landmarks (0,1)
    for i in [0, 1]:
        x, y = landmarks[i]
        cv2.circle(
            frame, (int(x), int(y)), 2, right_eye_color, -1, lineType=cv2.LINE_AA
        )

    # Draw left eye landmarks (2,3)
    for i in [2, 3]:
        x, y = landmarks[i]
        cv2.circle(frame, (int(x), int(y)), 2, left_eye_color, -1, lineType=cv2.LINE_AA)

    # Draw nose landmark (4)
    x, y = landmarks[4]
    cv2.circle(frame, (int(x), int(y)), 2, nose_color, -1, lineType=cv2.LINE_AA)


def segment_eyes(frame, landmarks, ow=160, oh=96):
    eyes = []

    # Segment eyes
    for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
        x1, y1 = landmarks[corner1, :]
        x2, y2 = landmarks[corner2, :]
        eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
        if eye_width == 0.0:
            return eyes

        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        # center image on middle of eye
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

        # center image
        center_mat = np.asmatrix(np.eye(3))
        center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
        inv_center_mat = np.asmatrix(np.eye(3))
        inv_center_mat[:2, 2] = -center_mat[:2, 2]

        # Get rotated and scaled, and segmented image
        transform_mat = center_mat * scale_mat * translate_mat
        inv_transform_mat = inv_translate_mat * inv_scale_mat * inv_center_mat

        eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (ow, oh))
        eye_image = cv2.equalizeHist(eye_image)

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
            assert gaze.shape == (2,)
            assert landmarks.shape == (34, 2)

            landmarks = landmarks * np.array([oh / 48, ow / 80])

            temp = np.zeros((34, 3))
            if eye.is_left:
                temp[:, 0] = ow - landmarks[:, 1]
            else:
                temp[:, 0] = landmarks[:, 1]
            temp[:, 1] = landmarks[:, 0]
            temp[:, 2] = 1.0
            landmarks = temp
            assert landmarks.shape == (34, 3)
            landmarks = np.asarray(np.matmul(landmarks, eye.transform_inv.T))[:, :2]
            assert landmarks.shape == (34, 2)
            result.append(EyePrediction(eye_sample=eye, landmarks=landmarks, gaze=gaze))
    return result


def spherical_to_cartesian(pitch, yaw):
    vx = np.cos(pitch) * np.sin(yaw)
    vy = np.sin(pitch)
    vz = np.cos(pitch) * np.cos(yaw)
    v = np.array([vx, vy, vz])
    v = v / np.linalg.norm(v)  # Normalize to get a unit vector
    return v


if __name__ == "__main__":
    main()
