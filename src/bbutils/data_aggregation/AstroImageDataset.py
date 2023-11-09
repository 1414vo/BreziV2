import glob
from astropy.io import fits


class AstroImageDataset():
    def __init__(self, observations_path, dark_path, flat_path):
        
        self.data = {}
        self.metadata = {}
        
        for file in glob.glob(f'{observations_path}/*.fits'):
            self.__store_fits(file)
                
        for file in glob.glob(f'{dark_path}/*.fits'):
            self.__store_fits(file, image_type='dark')
            
        for file in glob.glob(f'{observations_path}/*.fits'):
            self.__store_fits(file, image_type='flat_path')

    def __store_fits(self, file, image_type = 'observations'):
        if not image_type in self.data:
            self.data[image_type] = {}
            self.metadata[image_type] = {}
            
        with fits.open(file) as img:
            filename = file.split('/')[-1]
            self.metadata[image_type][filename] = img[0].header
            self.data[image_type][filename] = img[1].data
        
    def get_data(self):
        return self.data, self.metadata
    