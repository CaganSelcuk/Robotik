import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

focal_length = 1342.6866

marker_length = 0.025   

real_distances = [0.20, 0.25, 0.30, 0.35, 0.40]

image_folder = "lab5_foto"
image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))

if len(image_paths) == 0:
    print("No images found in 'lab5_foto' folder!")
    exit()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()

results = []

print("\nFound", len(image_paths), "images.\n")

for index, path in enumerate(image_paths):

    frame = cv2.imread(path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    print("Processing:", os.path.basename(path))

    if ids is None:
        print("  ❌ No ArUco marker detected.\n")
        continue


    corner = corners[0][0]
    pixel_width = np.linalg.norm(corner[0] - corner[1])


    distance = (marker_length * focal_length) / pixel_width


    real = real_distances[index] if index < len(real_distances) else None

    print(f"  Estimated distance: {distance:.3f} m")
    if real is not None:
        print(f"  Real distance:      {real:.3f} m")
        print(f"  Error:              {abs(distance - real):.3f} m\n")

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(f"Image {index+1}")
    plt.axis('off')
    plt.show()

    results.append((index + 1, distance, real))


print("\n===== LAB05 REPORT TABLE =====")
print("Image | ArUco Distance (m) | Real Distance (m)")
print("-----------------------------------------------")

for n, d, r in results:
    if r is None:
        print(f"{n}\t{d:.3f}\t\t\t -")
    else:
        print(f"{n}\t{d:.3f}\t\t\t {r:.3f}")