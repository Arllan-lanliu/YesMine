#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
___author__ = "Massimiliano Todisco, Hemlata Tak"
__email__ = "{todisco,tak}@eurecom.fr"
"""

'''
   Hemlata Tak, Madhu Kamble, Jose Patino, Massimiliano Todisco, Nicholas Evans.
   RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing.
   In Proc. ICASSP 2022, pp:6382--6386.
'''

import numpy as np
from scipy import signal
import copy


def rand_range(x1, x2, integer):
    """Generate random number in range [x1, x2]"""
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        y = int(y)
    return y


def norm_wav(x, always):
    """Normalize waveform"""
    if always:
        x = x / np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
        x = x / np.amax(abs(x))
    return x


def gen_notch_coeffs(n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g, fs):
    """Generate notch filter coefficients"""
    b = 1
    for i in range(0, n_bands):
        fc = rand_range(min_f, max_f, 0)
        bw = rand_range(min_bw, max_bw, 0)
        c = rand_range(min_coeff, max_coeff, 1)

        if c / 2 == int(c / 2):
            c = c + 1
        f1 = fc - bw / 2
        f2 = fc + bw / 2
        if f1 <= 0:
            f1 = 1 / 1000
        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000
        b = np.convolve(
            signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs), b
        )

    g = rand_range(min_g, max_g, 0)
    _, h = signal.freqz(b, 1, fs=fs)
    b = pow(10, g / 20) * b / np.amax(abs(h))
    return b


def filter_fir(x, b):
    """Apply FIR filter"""
    n = b.shape[0] + 1
    x_pad = np.pad(x, (0, n), 'constant')
    y = signal.lfilter(b, 1, x_pad)
    y = y[int(n / 2):int(y.shape[0] - n / 2)]
    return y


def lnl_convolutive_noise(
    x, n_f, n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g,
    min_bias_lin_nonlin, max_bias_lin_nonlin, fs
):
    """Linear and non-linear convolutive noise"""
    y = [0] * x.shape[0]
    for i in range(0, n_f):
        if i == 1:
            min_g = min_g - min_bias_lin_nonlin
            max_g = max_g - max_bias_lin_nonlin
        b = gen_notch_coeffs(
            n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g, fs
        )
        y = y + filter_fir(np.power(x, (i + 1)), b)
    y = y - np.mean(y)
    y = norm_wav(y, 0)
    return y


def isd_additive_noise(x, p, g_sd):
    """Impulsive signal dependent noise"""
    beta = rand_range(0, p, 0)

    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len * (beta / 100))
    positions = np.random.permutation(x_len)[:n]
    f_r = np.multiply(
        ((2 * np.random.rand(positions.shape[0])) - 1),
        ((2 * np.random.rand(positions.shape[0])) - 1)
    )
    r = g_sd * x[positions] * f_r
    y[positions] = x[positions] + r
    y = norm_wav(y, 0)
    return y


def ssi_additive_noise(
    x, snr_min, snr_max, n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff,
    min_g, max_g, fs
):
    """Stationary signal independent noise"""
    noise = np.random.normal(0, 1, x.shape[0])
    b = gen_notch_coeffs(
        n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g, fs
    )
    noise = filter_fir(noise, b)
    noise = norm_wav(noise, 1)
    snr = rand_range(snr_min, snr_max, 0)
    noise = (
        noise / np.linalg.norm(noise, 2) * np.linalg.norm(x, 2) / 10.0**(0.05 * snr)
    )
    x = x + noise
    return x


def process_rawboost_feature(feature, sr, args, algo):
    """Process feature using RawBoost augmentation"""
    # Data process by Convolutive noise (1st algo)
    if algo == 1:
        feature = lnl_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
            args.maxBiasLinNonLin, sr
        )

    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:
        feature = isd_additive_noise(feature, args.P, args.g_sd)

    # Data process by coloured additive noise (3rd algo)
    elif algo == 3:
        feature = ssi_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )

    # Data process by all 3 algo. together in series (1+2+3)
    elif algo == 4:
        feature = lnl_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
            args.maxBiasLinNonLin, sr
        )
        feature = isd_additive_noise(feature, args.P, args.g_sd)
        feature = ssi_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )

    # Data process by 1st two algo. together in series (1+2)
    elif algo == 5:
        feature = lnl_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
            args.maxBiasLinNonLin, sr
        )
        feature = isd_additive_noise(feature, args.P, args.g_sd)

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo == 6:
        feature = lnl_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
            args.maxBiasLinNonLin, sr
        )
        feature = ssi_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo == 7:
        feature = isd_additive_noise(feature, args.P, args.g_sd)
        feature = ssi_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )

    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo == 8:
        feature1 = lnl_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin,
            args.maxBiasLinNonLin, sr
        )
        feature2 = isd_additive_noise(feature, args.P, args.g_sd)

        feature_para = feature1 + feature2
        feature = norm_wav(feature_para, 0)  # normalized resultant waveform

    # original data without Rawboost processing
    else:
        feature = feature

    return feature