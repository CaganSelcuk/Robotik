import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import pipeline
from PIL import Image

# 0. CONSTANTS
FOCAL_LENGTH = 1342.69      # fx (pixels)
MARKER_LENGTH = 0.025       # 2.5 cm = 0.025 m

IMAGE_PATH = "images/lab6_aruco.jpg"

# 1. LOAD IMAGE
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# 2. DETECT ARUCO MARKERS WITH OPENCV
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, _ = detector.detectMarkers(gray)

if ids is None or len(ids) < 3:
    raise RuntimeError("Less than 3 ArUco markers were detected!")

ids = ids.flatten()
print(f"Detected marker IDs: {ids}")


opencv_dist = {}
marker_centers = {}

for i, marker_id in enumerate(ids):
    c = corners[i][0]  # 4×2
    width_pixels = np.linalg.norm(c[0] - c[1])
    distance = (MARKER_LENGTH * FOCAL_LENGTH) / width_pixels

    center = c.mean(axis=0)  # (x,y)

    opencv_dist[int(marker_id)] = float(distance)
    marker_centers[int(marker_id)] = center

    print(f"Marker ID {marker_id}: OpenCV distance = {distance:.3f} m, "
          f"center = [{center[0]:.2f}, {center[1]:.2f}]")

sorted_ids = sorted(opencv_dist.keys(), key=lambda mid: opencv_dist[mid])
print("\nSorted markers (near to far):")
for rank, mid in enumerate(sorted_ids, 1):
    print(f"Rank {rank}: ID={mid}, d_OpenCV={opencv_dist[mid]:.3f} m")

id_near, id_mid, id_far = sorted_ids  # first, second, third

# 3. DEPTH-ANYTHING V2 (RELATIVE & METRIC-LIKE MAPS)

print("\nLoading Depth-Anything-V2-Small (HF pipeline)...")
depth_pipe = pipeline(
    task="depth-estimation",
    model="pcuenq/Depth-Anything-V2-Small-hf",
    device="cpu"
)

pil_img = Image.fromarray(img_rgb)
depth_result = depth_pipe(pil_img)


depth_pil = depth_result["depth"]
depth_rel = np.array(depth_pil, dtype=np.float32)

d_min, d_max = depth_rel.min(), depth_rel.max()
depth_rel_norm = (depth_rel - d_min) / (d_max - d_min + 1e-8)

near_d = opencv_dist[id_near]
far_d = opencv_dist[id_far]

depth_metric = near_d + depth_rel_norm * (far_d - near_d)

# 4. DEPTH PER MARKER FROM DEPTH MAP
depth_anything_dist = {}

for mid in sorted_ids:
    cx, cy = marker_centers[mid]
    cx, cy = int(round(cx)), int(round(cy))

    x0, x1 = max(cx - 2, 0), min(cx + 3, depth_metric.shape[1])
    y0, y1 = max(cy - 2, 0), min(cy + 3, depth_metric.shape[0])

    patch = depth_metric[y0:y1, x0:x1]
    depth_mean = float(patch.mean())
    depth_anything_dist[mid] = depth_mean

    print(f"\nMarker ID {mid}:")
    print(f"  OpenCV distance         = {opencv_dist[mid]:.3f} m")
    print(f"  Depth-Anything distance = {depth_mean:.3f} m")

# 5. RATIO OF DISTANCES

d1_cv = opencv_dist[id_near]
d2_cv = opencv_dist[id_mid]
d3_cv = opencv_dist[id_far]

r_cv = (d2_cv - d1_cv) / (d3_cv - d2_cv + 1e-8)

d1_da = depth_anything_dist[id_near]
d2_da = depth_anything_dist[id_mid]
d3_da = depth_anything_dist[id_far]

r_da = (d2_da - d1_da) / (d3_da - d2_da + 1e-8)

print("\n===== RATIO OF DISTANCES (Lab 6 requirement) =====")
print(f"IDs (near, middle, far) = {id_near}, {id_mid}, {id_far}")
print(f"OpenCV  : (d2-d1)/(d3-d2) = {r_cv:.3f}")
print(f"DepthNet: (d2-d1)/(d3-d2) = {r_da:.3f}")

# 6. SHOW IMAGES
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Original image")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Relative depth (normalized)")
plt.imshow(depth_rel_norm, cmap="magma")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Metric-like depth (scaled)")
plt.imshow(depth_metric, cmap="viridis")
plt.axis("off")

plt.tight_layout()
plt.show()