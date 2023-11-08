from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

import numpy as np

import warnings

def __verify_input(input, transformation):
    if not transformation:
        raise NotFittedError("This MasterFlat instance has not been presented with any samples")
    
    if not 'observations' in X:
        raise ValueError('Input dictionary does not contain an "observation array"')
    
    if type(transformation) != float and transformation.shape != input['obervations'].shape[1:]:
        raise ValueError(f"Shape Mismatch: The images of size {input.shape[1:]} \
            should have the same shape as the applied image of size {transformation.shape}")
        
class MasterDark(BaseEstimator):
    def __init__(self):
        self.dark = None
    
    def fit(self, X, y):
        if not 'dark' in X:
            self.dark = 0.
            warnings.warn("No dark frames were provided, the end signal may contain significant noise.")
            
        if len(X) % 2 == 0:
            warnings.warn("The number of image samples should preferrably be an odd number to avoid outliers.")
            
        self.dark = np.median(X)
        
    def transform(self, X):
        
        __verify_input(X, self.dark)
        return X['observations'] - self.dark
    
class MasterFlat(BaseEstimator):
    def __init__(self):
        self.flat = None
    
    def fit(self, X, y):
        if not 'flat' in X:
            self.flat = 1.
            warnings.warn("No flat frames were provided, the end signal may contain innacuracies.")
            
        if len(X) % 2 == 0:
            warnings.warn("The number of image samples should preferrably be an odd number to avoid outliers.")
            
        self.flat = np.median(X)
        
    def transform(self, X):
        __verify_input(X, self.flat)
        return X['observations'] / self.dark