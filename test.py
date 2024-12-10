import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2
import json
import math
from matplotlib import pyplot as plt

from datasets.unity_eyes import UnityEyesDataset
from models.eyenet import EyeNet
from util.preprocess import gaussian_2d
from util.gaze import draw_gaze

device = torch.device("cpu")
img_dir = "/kaggle/input/eye-gaze-detection/TestSet"
json_dir = "/kaggle/input/eye-gaze-detection/TestSet_json"
dataset = UnityEyesDataset(img_dir=img_dir, json_dir=json_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

checkpoint = torch.load("checkpoint.pt", map_location=device)
nstack = checkpoint["nstack"]
nfeatures = checkpoint["nfeatures"]
nlandmarks = checkpoint["nlandmarks"]
eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
eyenet.load_state_dict(checkpoint["model_state_dict"])


def angular_error(gt_gaze, pred_gaze):
    # Normalize both vectors
    gt_norm = gt_gaze / np.linalg.norm(gt_gaze)
    pred_norm = pred_gaze / np.linalg.norm(pred_gaze)
    dot = np.clip(np.dot(gt_norm, pred_norm), -1.0, 1.0)
    angle = math.acos(dot)
    # Convert to degrees
    angle_degrees = angle * 180.0 / math.pi
    return angle_degrees


all_errors = []
num_plots_to_save = 5
plot_count = 0
output_plot_dir = "test_results"
os.makedirs(output_plot_dir, exist_ok=True)

with torch.no_grad():
    for i, sample in enumerate(dataloader):
        img_tensor = sample["img"].float().to(device).unsqueeze(0)  # [1,1,H,W]
        heatmaps_pred, landmarks_pred, gaze_pred = eyenet(img_tensor)
        landmarks_pred = landmarks_pred.cpu().numpy()[0, :]  # (34,2)
        gaze_pred = gaze_pred.cpu().numpy()[0, :]  # (2,)

        gt_gaze = sample["gaze"].numpy()[0]  # (2,)
        error = angular_error(gt_gaze, gaze_pred)
        all_errors.append(error)

        # If we still want to save some sample plots
        if plot_count < num_plots_to_save:
            # Create the predicted heatmaps as in the original code
            result = [
                gaussian_2d(w=80, h=48, cx=c[1], cy=c[0], sigma=3)
                for c in landmarks_pred
            ]

            img = sample["img"].numpy()[0]
            img = ((img - img.min()) / (img.max() - img.min()) * 255.0).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Ground truth iris center
            iris_center_gt = sample["landmarks"][0][-2][::-1] * 2
            img_gaze_gt = img.copy()
            cv2.circle(
                img_gaze_gt,
                (int(iris_center_gt[0]), int(iris_center_gt[1])),
                2,
                (0, 255, 0),
                -1,
            )
            draw_gaze(
                img_gaze_gt, iris_center_gt, gt_gaze, length=60, color=(0, 255, 0)
            )

            # Predicted iris center
            iris_center_pred = landmarks_pred[-2][::-1] * 2
            img_gaze_pred = img.copy()
            cv2.circle(
                img_gaze_pred,
                (int(iris_center_pred[0]), int(iris_center_pred[1])),
                2,
                (255, 0, 0),
                -1,
            )
            draw_gaze(
                img_gaze_pred, iris_center_pred, gaze_pred, length=60, color=(255, 0, 0)
            )

            # Gaze comparison figure
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(cv2.cvtColor(img_gaze_gt, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Ground Truth Gaze")
            axes[0].axis("off")

            axes[1].imshow(cv2.cvtColor(img_gaze_pred, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f"Predicted Gaze (Error: {error:.2f}°)")
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(output_plot_dir, f"gaze_comparison_{i}.png"))
            plt.close(fig)

            # Heatmap comparison figure
            heatmaps = sample["heatmaps"].numpy()[0]  # (34,H,W)
            img_gaze_orig = img.copy()
            img_gaze_gt_gray = cv2.cvtColor(img_gaze_gt, cv2.COLOR_BGR2GRAY)
            img_gaze_pred_gray = cv2.cvtColor(img_gaze_pred, cv2.COLOR_BGR2GRAY)

            fig2 = plt.figure(figsize=(8, 9))

            # Raw training image
            plt.subplot(321)
            full_img = sample["full_img"].numpy()[0]
            full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
            plt.imshow(full_img)
            plt.title("Raw training image")
            plt.axis("off")

            # Preprocessed training image
            plt.subplot(322)
            plt.imshow(cv2.cvtColor(img_gaze_orig, cv2.COLOR_BGR2GRAY), cmap="gray")
            plt.title("Preprocessed training image")
            plt.axis("off")

            # Ground truth heatmaps
            plt.subplot(323)
            plt.imshow(np.mean(heatmaps[16:32], axis=0), cmap="gray")
            plt.title("Ground truth heatmaps")
            plt.axis("off")

            # Predicted heatmaps
            plt.subplot(324)
            plt.imshow(np.mean(result[16:32], axis=0), cmap="gray")
            plt.title("Predicted heatmaps")
            plt.axis("off")

            # Ground truth landmarks and gaze vector
            plt.subplot(325)
            plt.imshow(img_gaze_gt_gray, cmap="gray")
            plt.title("Ground truth landmarks and gaze vector")
            plt.axis("off")

            # Predicted landmarks and gaze vector
            plt.subplot(326)
            plt.imshow(img_gaze_pred_gray, cmap="gray")
            plt.title("Predicted landmarks and gaze vector")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(output_plot_dir, f"heatmap_comparison_{i}.png"))
            plt.close(fig2)

            plot_count += 1

# Compute average gaze angular error
avg_error = np.mean(all_errors)
print(f"Average Gaze Angular Error: {avg_error:.2f} degrees")

# Save evaluation results to text file
results_txt_path = os.path.join(output_plot_dir, "evaluation_results.txt")
with open(results_txt_path, "w") as f:
    f.write("Index,AngularError\n")
    for idx, err in enumerate(all_errors):
        f.write(f"{idx},{err}\n")
    f.write(f"Average Error: {avg_error}\n")

print(f"Evaluation results and plots saved in {output_plot_dir}")
print(f"Results saved to {results_txt_path}")
