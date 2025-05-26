from .base import BaseLogger
import pickle
import os

class MoleculeLogger(BaseLogger):

    def __init__(self, args, output_formats):
        super(MoleculeLogger, self).__init__(args, output_formats)
        
    def log_samples(self, molecule_objs, fname='molecules'):
        name = os.path.join(self.logging_dir, f'{fname}.pkl')
        with open(name, 'wb') as f:
            pickle.dump(molecule_objs, f)

        super(MoleculeLogger, self).log_samples(None)
    
    def load_samples(self, fname="molecules"):
        name = os.path.join(self.logging_dir, f'{fname}.pkl')
        with open(name, 'rb') as f:
            obj = pickle.load(f)
        return obj