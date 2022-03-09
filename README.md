# EEG Biometrics
Library for EEG Biometrics project for HERO Lab - UC Irvine

![This is an image](https://cdn.freelogovectors.net/wp-content/uploads/2019/09/uci-logo.png)

# Requirements

- `Python` == 3.7 or 3.8
- `tensorflow` == 2.X (verified working with 2.0 - 2.3, both for CPU and GPU)
- `keras` >= 2.8.0

- `mne` >= 0.24.1
- `librosa` >= 0.9.1
- `sklearn` >= 1.0.2
- `scikit-learn` >= 0.20.1
- `matplotlib` >= 2.2.3
- `pandas` >=  1.4.1
- `numpy` >= 1.22.0

# Packages

## prep

Python module that implements the PrePipeline algorithm by epoching a raw eeg signals

## ppeeg

Python module with some EEG preprocessing functions for biometrics applications
 
This module requires some libraries:
- `pyprep` (Preprocessing Pipeline): Standardized preprocessing for large-scale EEG analysis. DOI: 10.3389/fninf.2015.00016
- `MNE`: Open-source Python package for exploring, visualizing, and analyzing human neurophysiological data: MEG, EEG, sEEG, ECoG, NIRS, and more.
- `Librosa`: Python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems.

## pyOpenBCI

Python module for interfacing with the OpenBCI devices.

This module requires `Serial`

## RaspberryPiADS1299

Python module for interfacing with host device and the ADS1299 (Analog-to-Digital Converter for Biopotential Measurements)

This module requires some libraries:
- `spidev`: Python module for interfacing with SPI devices from user space via the spidev linux kernel driver.

Ref: https://github.com/wjcroft/RaspberryPiADS1299

## ClassifiersModelsEEG

Python module with different Deep Learning Classifiers models for EEG biometrics

This module requires `keras` and `tensorflow`

## ClassifierEEG

Python module that makes the classification for the EEG samples read from the ADS1299 

## utilsEEG

Python module with some usefull tools for the classification models

# main

Main script of the EEG Biometrics device. Use the OpenBCI headset to collect EEG signals from a subject and by the way of different tools and Deep Learning models, predict the person identity.
