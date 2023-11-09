from sklearn.base import BaseEstimator
import numpy as np
import sys
sys.path.append('../')
from imaging_utils.imaging_utils import get_image_groups
class LightCurveExtractor(BaseEstimator):
    
    def __init__(self, object_mask, standard_mask):
        self.object_mask = object_mask
        self.mask_groups = get_image_groups(self.standard_mask)
        
    def fit(self, *args, **kwargs):
        pass

    def transform(self, X, *args, **kwargs):
        curve = []
        standard_curves = [[] for _ in len(self.mask_groups)]
        for img in X['observations']:
            if self.object_mask.shape != img.shape:
                raise ValueError(f"Mask ({self.object_mask.shape}) should have the same shape as images ({img.shape})")
            curve.append(np.sum(self.object_mask * img))
            
            for i, mask in enumerate(self.mask_groups):
                if self.standard_mask.shape != img.shape:
                    raise ValueError(f"Mask ({mask.shape}) should have the same shape as images ({img.shape})")
                standard_curves[i].append(np.sum(mask * img))
        return np.array(curve)