import numpy as np
import pandas as pd


import mne
from EEG_Biometrics.pyprep.prep_pipeline import PrepPipeline
import librosa.util as lu
import librosa.feature as lf

class PreProcessingEEG(object):
           
    def PREP(samples):
        samples_info = mne.create_info(ch_names="AF3 T7 P7 O1 O2 P8 T8 AF4".split(), sfreq=250, ch_types=["eeg"]*8)
        raw = mne.io.RawArray(samples, samples_info)
        
        # The eegbci data has non-standard channel names. We need to rename them:
        mne.datasets.eegbci.standardize(raw)

        # Add a montage to the data
        montage_kind = "standard_1005"
        montage = mne.channels.make_standard_montage(montage_kind)

        # Extract some info
        sample_rate = raw.info["sfreq"]

        # Make a copy of the data
        raw_copy = raw.copy()
        
        # Fit prep
        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": np.arange(1, sample_rate / 2, 1),
        }
        
        prep = PrepPipeline(raw_copy, prep_params, montage)
        prep.fit()
        return raw_copy.get_data(picks="eeg") * 1e6


    def extract_bands(raw):
        # let's explore some frequency bands
        iter_freqs = [
            ('Theta', 4, 7),
            ('Alpha', 8, 12),
            ('Beta', 13, 25),
            ('Gamma', 30, 45)
        ]
        # set epoching parameters
        event_id, tmin, tmax = 1, -1., 3.
        baseline = None
        events = mne.find_events(raw)
        
        frequency_map = list()
        
        for band, fmin, fmax in iter_freqs:
            # bandpass filter
            raw.filter(fmin, fmax, n_jobs=1,  # use more jobs to speed up.
                       l_trans_bandwidth=1,  # make sure filter params are the same
                       h_trans_bandwidth=1)  # in each band and skip "auto" option.

            # epoch
            epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=baseline,
                                reject=dict(grad=4000e-13, eog=350e-6),
                                preload=True)
            # remove evoked response
            epochs.subtract_evoked()

            # get analytic signal (envelope)
            epochs.apply_hilbert(envelope=True)
            frequency_map.append(((band, fmin, fmax), epochs.average()))

        return frequency_map
        
    
    
    def extract_features(frequency_map):
        ## sc_band1_ch1 | sb_band1_ch1 | scf_band1_ch1 | sf_band1_ch1 | sc_band2_ch1 | sb_band2_ch1 | scf_band2_ch1 | sf_band2_ch1 ....  sc_band4_ch14 | sb_band4_ch14 | scf_band4_ch14 | sf_band4_ch14
        
        header = []
        for ch in range(1,15):
            for band in ['Theta', 'Alpha', 'Beta', 'Gamma']:
                for f in ['SC', 'SB', 'SCF', 'SF']:
                    header.append('{}_{}_CH{}'.format(f, band, str(ch)))
        
        extracted_features = []
        for channel in frequency_map[0][1].info['ch_names']:
            for band in frequency_map:
                band_info = band[0]
                signal = band[1].pick_channels([channel])
            
                spectral_centroid = lf.spectral.spectral_centroid(y=signal.get_data()[0], sr=signal.info['sfreq'], hop_length=1024)[0][0]
                extracted_features.append(spectral_centroid)
                
                spectral_bandwidth = lf.spectral.spectral_bandwidth(y=signal.get_data()[0], sr=signal.info['sfreq'], hop_length=1024)[0][0]
                extracted_features.append(spectral_bandwidth)
                
                peaks = signal.get_data()[0][lu.peak_pick(signal.get_data()[0], pre_max=0, post_max=1, pre_avg=0, post_avg=1, delta=0, wait=999)[0]]
                rms = lf.spectral.rms(y=signal.get_data()[0], hop_length=1024)[0][0]
                spectral_crest_factor = peaks/rms
                extracted_features.append(spectral_crest_factor)
                
                spectral_flatness = lf.spectral.spectral_flatness(y=signal.get_data()[0], hop_length=1024)[0][0]
                extracted_features.append(spectral_flatness)
        
        return extracted_features