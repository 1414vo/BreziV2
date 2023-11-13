from sklearn.base import BaseEstimator
from ..imaging_utils import imaging_utils
class Aligner(BaseEstimator):
    def __init__(self):
        self.reference = None
        
    def fit(self, X):
        self.reference = X['observations'][0]
        
    def transform(self, X):
        for i, img in enumerate(X['observations']):
            X['observations'][i] = imaging_utils.register_images(img, self.reference)
            
        return X