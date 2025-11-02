import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMAGE_PATH = "1704513.jpg"
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"{IMAGE_PATH} not found!")

blurred = cv2.GaussianBlur(image, (5, 5), 0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
edges = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))
edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]], dtype=np.float32)
sharpened = cv2.filter2D(image, -1, kernel)

kernel_emboss = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]], dtype=np.float32)
embossed = cv2.filter2D(image, -1, kernel_emboss)

combined = cv2.addWeighted(blurred, 0.5, edges_bgr, 0.5, 0)
combined = cv2.addWeighted(combined, 0.5, sharpened, 0.5, 0)

plt.figure(figsize=(10, 7))

plt.subplot(2, 3, 1)
plt.title('Оригинальное изображение')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Размытие по Гауссу')
plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Выделение границ')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Повышение резкости')
plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Комбинация изображений')
plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title('Собственный фильтр (Emboss)')
plt.imshow(cv2.cvtColor(embossed, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()

output_dir = Path("lab1_outputs")
output_dir.mkdir(exist_ok=True)

plt.savefig(output_dir / "lab1_result_grid.png", dpi=200)
cv2.imwrite(str(output_dir / "00_original.png"), image)
cv2.imwrite(str(output_dir / "01_gaussian_blur.png"), blurred)
cv2.imwrite(str(output_dir / "02_edges_sobel.png"), edges)
cv2.imwrite(str(output_dir / "03_sharpened.png"), sharpened)
cv2.imwrite(str(output_dir / "04_combined.png"), combined)
cv2.imwrite(str(output_dir / "05_custom_emboss.png"), embossed)

print("Saved results in folder:", output_dir.resolve())
plt.show()