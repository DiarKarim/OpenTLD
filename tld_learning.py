import numpy as np
import cv2

class NearestNeighborModel:
    def __init__(self, ncc_threshold=0.95):
        self.positive_examples = []  # List of np.ndarray
        self.negative_examples = []  # List of np.ndarray
        self.ncc_threshold = ncc_threshold

    def add_positive(self, patch):
        self.positive_examples.append(patch)

    def add_negative(self, patch):
        self.negative_examples.append(patch)

    def similarity(self, patch):
        # Compute NCC to all positive and negative examples
        if not self.positive_examples:
            return 0.0, 0.0, [np.nan, np.nan, np.nan]
        if not self.negative_examples:
            return 1.0, 1.0, [np.nan, np.nan, np.nan]
        
        patch = patch.flatten()
        pos_ncc = [normalized_cross_correlation(patch, p) for p in self.positive_examples]
        neg_ncc = [normalized_cross_correlation(patch, n) for n in self.negative_examples]
        
        isin = [0, np.argmax(pos_ncc), 0]
        if np.any(np.array(pos_ncc) > self.ncc_threshold):
            isin[0] = 1
        if np.any(np.array(neg_ncc) > self.ncc_threshold):
            isin[2] = 1
        
        dN = 1 - np.max(neg_ncc)
        dP = 1 - np.max(pos_ncc)
        conf1 = dN / (dN + dP)
        conf2 = conf1  # For simplicity, no conservative subset
        return conf1, conf2, isin

    def update(self, positive_patches, negative_patches, thr_nn=0.65):
        # Add new positive examples if not similar enough
        for patch in positive_patches:
            conf1, _, isin = self.similarity(patch)
            if conf1 <= thr_nn:
                self.add_positive(patch)
        # Add new negative examples if not already negative
        for patch in negative_patches:
            conf1, _, _ = self.similarity(patch)
            if conf1 > 0.5:
                self.add_negative(patch)

def extract_patch(img, bbox, patch_size):
    # bbox: [x, y, w, h]
    x, y, w, h = map(int, bbox)
    patch = img[y:y+h, x:x+w]
    patch = cv2.resize(patch, patch_size)
    return normalize_patch(patch)

def normalize_patch(patch):
    # Zero mean, unit variance
    patch = patch.astype(np.float32)
    patch = patch.flatten()
    mean = np.mean(patch)
    std = np.std(patch) if np.std(patch) > 0 else 1.0
    return (patch - mean) / std

def normalized_cross_correlation(a, b):
    # Both a and b should be 1D arrays
    a = a.flatten()
    b = b.flatten()
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def tld_learning_step(nn_model, img, bbox, patch_size, positive_bboxes, negative_bboxes, thr_nn=0.65):
    # Extract current patch
    current_patch = extract_patch(img, bbox, patch_size)
    # Consistency checks (variance, similarity, etc.) can be added here
    # Generate positive and negative patches
    positive_patches = [extract_patch(img, bb, patch_size) for bb in positive_bboxes]
    negative_patches = [extract_patch(img, bb, patch_size) for bb in negative_bboxes]
    # Update the nearest neighbor model
    nn_model.update(positive_patches, negative_patches, thr_nn=thr_nn)
    return nn_model 