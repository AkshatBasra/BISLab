import numpy as np
from skimage import data, img_as_float, transform
import matplotlib.pyplot as plt

# --- Step 1: Load grayscale image (already grayscale, no conversion needed) ---
img = data.coins()                    # grayscale image
# img = transform.resize(img, (80, 80), anti_aliasing=True)
img = img_as_float(img)
height, width = img.shape

# --- Step 2: Parameters ---
n_clusters = 3
max_iter = 50
mutation_rate = 0.05

# --- Step 3: Initialize random labels ---
labels = np.random.randint(0, n_clusters, (height, width))

# --- Step 4: Define neighborhood ---
neighbors = [(-1,0), (1,0), (0,-1), (0,1)]

# --- Step 5: Fitness computations ---
def compute_cluster_means(labels):
    means = np.zeros(n_clusters)
    for k in range(n_clusters):
        cluster_pixels = img[labels == k]
        means[k] = np.mean(cluster_pixels) if cluster_pixels.size > 0 else 0
    return means

def compute_fitness(labels, means):
    fit_intensity = 1 - np.abs(img - means[labels])

    smooth = np.zeros_like(labels, dtype=float)
    for dy, dx in neighbors:
        shifted = np.roll(np.roll(labels, dy, axis=0), dx, axis=1)
        smooth += (shifted == labels)
    fit_smooth = smooth / len(neighbors)

    return 0.5 * fit_intensity + 0.5 * fit_smooth

# --- Step 6: Show initial images ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(labels, cmap='nipy_spectral')
plt.title("Initial Random Labels")
plt.axis('off')
plt.show()

# --- Step 7: Evolution loop ---
for it in range(max_iter):
    means = compute_cluster_means(labels)
    fitness = compute_fitness(labels, means)

    dy, dx = neighbors[np.random.randint(len(neighbors))]
    neighbor_labels = np.roll(np.roll(labels, dy, axis=0), dx, axis=1)

    # Crossover + mutation
    child_labels = np.where(np.random.rand(height, width) < 0.5, labels, neighbor_labels)
    mutation_mask = np.random.rand(height, width) < mutation_rate
    child_labels[mutation_mask] = np.random.randint(0, n_clusters, np.count_nonzero(mutation_mask))

    # Evaluate and replace
    child_means = compute_cluster_means(child_labels)
    child_fitness = compute_fitness(child_labels, child_means)
    labels = np.where(child_fitness > fitness, child_labels, labels)

# --- Step 8: Show final segmentation ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(labels, cmap='nipy_spectral')
plt.title("Final Segmentation (PCA Result)")
plt.axis('off')
plt.show()
