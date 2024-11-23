import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_edges(image, kernel, low_thresh=20, high_thresh=50):
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel, _, _ = cv2.split(ycbcr_image)

    convolved = cv2.filter2D(y_channel, ddepth=cv2.CV_64F, kernel=kernel)
    
    p = 1
    vertical_diff = np.diff(np.sign(convolved), axis=0)
    horizontal_diff = np.diff(np.sign(convolved), axis=1)
    
    zero_crossings = np.zeros_like(y_channel, dtype=bool)
    zero_crossings[:-1, :] |= (abs(vertical_diff) > p)
    zero_crossings[:, :-1] |= (abs(horizontal_diff) > p)
    
    edges = np.zeros_like(y_channel, dtype=np.uint8)
    for i in range(y_channel.shape[0]):
        for j in range(y_channel.shape[1]):
            if zero_crossings[i, j]:
                y1 = i + 1
                y2 = i - 1
                x1 = j + 1
                x2 = j - 1
                dx = 0
                dy = 0
                if y1 > y_channel.shape[0] - 1:
                    dy = y_channel[y2, j] - y_channel[i, j]
                elif y2 < 0:
                    dy = y_channel[y1, j] - y_channel[i, j]
                else:
                    dy = y_channel[y2, j] - y_channel[y1, j]
                if x1 > y_channel.shape[1] - 1:
                    dx = y_channel[i, x2] - y_channel[i, j]
                elif x2 < 0:
                    dx = y_channel[i, x1] - y_channel[i, j]
                else:
                    dx = y_channel[i, x1] - y_channel[i, x2]
                edges[i, j] = np.sqrt(dx**2 + dy**2)

    strong_edges = (edges > high_thresh)
    weak_edges = (edges > low_thresh) & ~strong_edges

    final_edges = np.zeros_like(y_channel, dtype=np.uint8)
    final_edges = final_edges + 255
    final_edges[strong_edges] = 0

    for i in range(y_channel.shape[0]):
        for j in range(y_channel.shape[1]):
            if weak_edges[i, j] and np.any(strong_edges[max(0, i-1):min(y_channel.shape[0], i+2), max(0, j-1):min(y_channel.shape[1], j+2)]):
                final_edges[i, j] = 255

    return final_edges

laplacian_kernel = np.array([[0, 0, 1, 0, 0],
                             [0, 1, 2, 1, 0],
                             [1, 2, -16, 2, 1],
                             [0, 1, 2, 1, 0],
                             [0, 0, 1, 0, 0]])

image_path = "cartoon_pic.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Please check the file path.")
print(image.shape)
edges_y = detect_edges(image, laplacian_kernel, low_thresh=1, high_thresh=1.5)

plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.title("Original Color Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Detected Edges (Y Channel in YCbCr)")
plt.imshow(edges_y, cmap='gray')
plt.axis('off')

plt.show()
