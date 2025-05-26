import torch
import librosa
import numpy as np

from .base import BaseEvaluator
from diffusion.audio_diffusion import AudioDiffusionSampler
from tasks.utils import load_audio_dataset
from utils.env_utils import *
from .utils.frechet_audio_distance import FrechetAudioDistance

import logger
import os
import scipy

class AudioEvaluator(BaseEvaluator):

    def __init__(self, args):
        super(AudioEvaluator, self).__init__()

        self.args = args

    def normalize(self, samples, ref_samples):
        max_abs_ref = torch.abs(ref_samples).max()
        max_abs_samples = torch.abs(samples).max()
        return samples / max_abs_samples * max_abs_ref, ref_samples

    def _compute_recover_score(self, samples, dataset):
        
        _, ref_audios = load_audio_dataset(dataset, len(samples))
        samples, ref_audios = self.normalize(samples, ref_audios)

        samples_mfcc = librosa.feature.mfcc(y=samples.numpy(), sr=self.args.sample_rate)
        ref_mfcc = librosa.feature.mfcc(y=ref_audios.numpy(), sr=self.args.sample_rate)

        cost = np.mean([librosa.sequence.dtw(samples_mfcc[i], ref_mfcc[i])[0][-1][-1] for i in range(ref_mfcc.shape[0])])

        return - cost / 1000

    def _compute_fad(self, samples, dataset):

        _, ref_audios = load_audio_dataset(dataset, len(samples))
        samples, ref_audios = self.normalize(samples, ref_audios)

        frechet = FrechetAudioDistance(
            model_name="vggish",
            sample_rate=16000,
            use_pca=False, 
            use_activation=False,
            verbose=False
        )
        
        resampled_ref_audios = librosa.resample(ref_audios.numpy(), orig_sr=self.args.sample_rate, target_sr=16000)
        resampled_audios = librosa.resample(samples.numpy(), orig_sr=self.args.sample_rate, target_sr=16000)

        embds_background = frechet.get_embeddings(resampled_ref_audios, self.args.sample_rate)
        embds_generated = frechet.get_embeddings(resampled_audios, self.args.sample_rate)
        # Compute statistics and FAD score
        mu_background, sigma_background = frechet.calculate_embd_statistics(embds_background)
        mu_eval, sigma_eval = frechet.calculate_embd_statistics(embds_generated)
        
        fad_score = frechet.calculate_frechet_distance(
            mu_background,
            sigma_background,
            mu_eval,
            sigma_eval,
            device=self.args.device
        )

        return fad_score

    def evaluate(self, samples):
        
        metrics = {}

        samples = AudioDiffusionSampler.obj_to_tensor(samples)

        logger.log(f"Evaluating {len(samples)} samples")

        recover_score = self._compute_recover_score(samples, self.args.dataset)
        metrics['recover_score'] = recover_score

        fad = self._compute_fad(samples, self.args.dataset)
        metrics['fad'] = fad
            
        return metrics