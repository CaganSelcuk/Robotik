import cv2
import matplotlib.pyplot as plt
from pathlib import Path

IMAGE_PATH = "face.jpg"
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"{IMAGE_PATH} not found!")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
inverted_edges = 255 - edges

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Оригинальное изображение')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Фильтр Canny (границы)')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Графический рисунок')
plt.imshow(inverted_edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
output_dir = Path("lab2_outputs"); output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "lab2_canny_self.png", dpi=200)
plt.show()