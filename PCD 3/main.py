
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt

gambar = iio.imread("reemar.jpg")

if gambar.ndim == 3:
    grayscale = np.dot(
        gambar[..., :3],
        [0.299, 0.587, 0.114]
    ).astype(np.uint8)
else:
    grayscale = gambar.astype(np.uint8)

histogram = np.bincount(grayscale.ravel(), minlength=256)

print("Jumlah total piksel:", grayscale.size)
print("\nHistogram Intensitas Grayscale:")

for intensitas in range(256):
    print(f"Intensitas {intensitas}: {histogram[intensitas]} piksel")

dominan = np.argmax(histogram)

print("\nIntensitas paling dominan:", dominan)
print("Jumlah piksel:", histogram[dominan])

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(gambar)
plt.title("Gambar RGB")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(grayscale, cmap="gray", vmin=0, vmax=255)
plt.title("Gambar Grayscale")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.plot(range(256), histogram)
plt.title("Histogram Grayscale")
plt.xlabel("Intensitas Piksel (0-255)")
plt.ylabel("Jumlah Piksel")
plt.xlim(0, 255)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
