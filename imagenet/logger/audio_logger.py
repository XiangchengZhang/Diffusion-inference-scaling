from .base import BaseLogger
import pickle
import os
import scipy
import torchaudio

class AudioLogger(BaseLogger):

    def __init__(self, args, output_formats):
        super(AudioLogger, self).__init__(args, output_formats)
        
    def log_samples(self, audio_objs, fname='audio'):

        audio_dir = os.path.join(self.logging_dir, 'audios')
        os.makedirs(audio_dir, exist_ok=True)

        self.log(f'audio samples are saved in {audio_dir}')

        for i, audio_obj in enumerate(audio_objs):
            name = os.path.join(audio_dir, f'{fname}_{i}.wav')
            scipy.io.wavfile.write(name, rate=audio_obj[1], data=audio_obj[0])
    
    def load_samples(self, fname='audios'):
        return None