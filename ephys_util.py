from glob import glob
import inspect
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, sosfreqz
import seaborn as sns


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    From https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html

    Might be a better way to implement here that uses second-order sections:
    https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=1):
    """
    From https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data, axis=axis)
    return y

def load_wm(filename):
    """
    Load .bin file from white matter electophysiology system.

    Parameters:
    filenme: str
        Path to .bin file

    Returns:
    sr: int
        Sampling rate of physiology experiment.
    data: 2-D array
        (n_samples, n_channels) numpy array.
    """

    #parse filename to get number of channels
    n_channels = int(filename.split('_')[-2][:-2])

    #load in binary data
    _data = np.fromfile(filename,'int16', offset=8)

    #reshape data to (n_channels, n_samples) and scale values to MICROVOLTS
    # Transpose to row-major indexing
    data = np.multiply(_data.reshape(-1, n_channels).T, 6.25e3/32768, dtype=np.float32)

    #parse filename to get sampling rate
    sr = int(filename.split('_')[-1][:-7])

    return sr, data


def truncate_audio(analog_file, data_phys, combine_audio = False):

    """
    Truncate beginning and end of audio to be aligned with physiology data and same length (in seconds)

    Parameters:
    exp_dir: str
        directory where phys and analog data live

    data_phys: 2-D array
        output from load_wm() - a 2-D array that is (n_samples x n_channels)

    combine_audio: boolean
        if True, function ouputs a single 1-D audio array with signal average from all mics.
        if False, function outputs N 1-D audio arrays for each mic.

    Returns:
    audio: N-D array

    if combine_audio True:
        audio: 1-D array:
            average audio signal

    if combine_audio False:
        mic1, mic2: 2-D array:
            audio signals

    """

    #load analog file
    data_analog = h5py.File(analog_file, 'r')

    # load the ephys trigger sample number (this is where we will truncate the data at the beginning)
    ephys_trigger_rising_edge = data_analog['analog_input'].attrs['ephys_trigger_rising_edge']
    #TODO return the rising trigger edge
    print('Ephys trigger detected at analog sample number {}'.format(ephys_trigger_rising_edge))
    print()

    #calculate the end truncation sample number
    #TODO: don't hard code the analog/ephys sampling ratio
    sampling_rate_ratio = 10
    end_trunc = (len(data_phys)*sampling_rate_ratio)

    if combine_audio == True:
        _mic1 = data_analog['analog_input'][0]
        mic1 = _mic1[ephys_trigger_rising_edge:ephys_trigger_rising_edge+end_trunc]

        _mic2 = data_analog['analog_input'][1]
        mic2 = _mic2[ephys_trigger_rising_edge:ephys_trigger_rising_edge+end_trunc]

        audio = 0.5 * (mic1 + mic2)

        # I think this line (np.array([audio])) intended to create an array of 2 dimensions
        return audio.reshape((1, -1)), ephys_trigger_rising_edge

    else:
        _mic1 = data_analog['analog_input'][0]
        mic1 = _mic1[ephys_trigger_rising_edge:ephys_trigger_rising_edge+end_trunc]

        _mic2 = data_analog['analog_input'][1]
        mic2 = _mic2[ephys_trigger_rising_edge:ephys_trigger_rising_edge+end_trunc]

        return np.stack((mic1, mic2), axis=0), ephys_trigger_rising_edge


    data_analog.close()


def load_data(exp_dir, phys_bandpass=(200, 2000), combine_audio=True):
    """
        Parameters:
        exp_dir: str
            directory where phys and analog data live
        phys_bandpass: tuple or boolean
            range (in Hz) to bandpass phys data. If False, raw data returned.
    """

        #search the exp directory for the physiology and analog data
    wm_file = glob(os.path.join(exp_dir, '*.bin'))[0]
    analog_file = glob(os.path.join(exp_dir, '*.h5'))[0]

    #load white matter data and check for bandpass argument...may take a while
    if phys_bandpass == False:
        print('Loading physiology data...')
        sr_phys, data_phys = load_wm(wm_file)
        print('No bandpass filtering.')
        print('Computing mean signal across all channels...')
        # Add a new axis where 64 used to be to allow numpy to broadcast the subtraction
        data_phys_mean = np.mean(data_phys, axis=0).reshape((1, -1))
        print('Subtracting mean signal across all channels...')
        data_phys = data_phys - data_phys_mean
        print()

    else:
        print('Loading physiology data...')
        sr_phys, data_phys = load_wm(wm_file)
        print('Bandpassing from {}-{} Hz...'.format(phys_bandpass[0], phys_bandpass[1]))
        data_phys_filt = butter_bandpass_filter(data_phys, phys_bandpass[0], phys_bandpass[1], sr_phys)
        print('Computing mean signal across all channels...')
        # Add a new axis where 64 used to be to allow numpy to broadcast the subtraction
        data_phys_mean = np.mean(data_phys_filt, axis=0).reshape((1, -1))
        print('Subtracting mean signal across all channels...')
        data_phys = data_phys_filt - data_phys_mean
        print()

    #load and truncate audio data
    print('Truncating audio...')
    data_audio, ephys_trigger_rising_edge = truncate_audio(analog_file, data_phys, combine_audio = combine_audio)
    print()

    return sr_phys, data_phys, data_audio, ephys_trigger_rising_edge


def get_spikes(data_ephys, threshold = 4):

    """
    Quick and dirty thresholding of phsyiology data to extract spikes.

    Parameters:
    data_ephys: 2-D array
        Physiology data. Expects (N_channel x N_sample) array
    threshold: int
        How many standard deviations below the mean to draw the threshold

    Returns:
    spikes: 2-D array
        (N_channels, N_spikes) array where values in N_spikes == spike times (in units of samples)

    """

    # Everything is broken up into subfunctions to get their individual runtimes

    # Calculate the threshold for every channel at the same time
    chan_thresholds = np.mean(data_ephys, axis=1) - np.std(data_ephys, axis=1) * threshold

    # The threshold comparison can also be done all at once:
    # Similar to my rising edge detector, looks for places where
    # arr[i] = False and arr[i+1] = True
    mask = data_ephys < chan_thresholds.reshape((data_ephys.shape[0], 1))
    spike_locations_2d = np.nonzero(~(mask[:,:-1]) & (mask[:,1:]))

    # The index search can alse be done all at once:
    ''' Spike_locations_2d[0] is an ndarray containing the 0-dim (channel number) for each spike.
    spike_locations_2d[1] has the 1-dim (index number within the channel'''
    abscissa_start = spike_locations_2d[0].searchsorted(np.arange(data_ephys.shape[0]), side='left')
    abscissa_end = spike_locations_2d[0].searchsorted(np.arange(data_ephys.shape[0]), side='right')

    # In the event there are no spikes in a channel, the start and end will be the same number, so a view into
    # an empty ndarray is returned
    return [spike_locations_2d[1][start:end] for start, end in zip(abscissa_start, abscissa_end)]


def truncate_spikes(spikes, onset, offset, ephys_trigger,sr_audio=125000, sr_phys=12500, n_channels=64):

    """
    Parameters:
    TODO

    Returns:
    spikes truncated between the onset/offset input
    """
    onset_audio = int(onset*sr_audio) - ephys_trigger #this is where in the truncated audio the sound event beings
    offset_audio = int(offset*sr_audio) - ephys_trigger #this is where in the truncated audio the sound event ends

    onset_phys = int(onset_audio/10)
    offset_phys = int(offset_audio/10)

    spikes_trunc = []

    # TODO: replace len(spikes) with n_channels for clarity?
    for i in range(n_channels):  # TODO: make this run in O(mlog(N))
        # Assume spikes[i] is ordered
        start_idx = spikes[i].searchsorted(onset_phys)
        end_idx = spikes[i].searchsorted(offset_phys, side='right')
        if end_idx > start_idx and spikes[i][end_idx] > offset_phys:
            end_idx -= 1

        working_spikes_trunc = spikes[i][start_idx:end_idx] - onset_phys
        spikes_trunc.append(working_spikes_trunc)

    # TODO: next benchmark, replace obj ndarray with list of ndarray
    return spikes_trunc

def psth(spikes,data_audio, onset_s, offset_s, ephys_trigger,
    pad = 1, hist_binsize=0.05, sr_audio=125000, sr_phys=12500, spec_clim=(-100,-70)):

    """
    Shows raw audio, psth, and spike raster for all channels.

    Parameters:
    TODO

    Returns:
    plot
    """
    onset = onset_s - pad
    offset = offset_s + pad

    onset_audio, offset_audio = int(onset*sr_audio)-ephys_trigger, int(offset*sr_audio)-ephys_trigger
    audio = data_audio[0, onset_audio:offset_audio]

    spikes_trunc = truncate_spikes(spikes,
                        onset = onset,
                        offset = offset,
                        ephys_trigger=ephys_trigger)


    plt.figure(figsize=(12,10))

    plt.subplot(411)
    plt.specgram(audio, NFFT=512, noverlap=256, Fs=sr_audio, xextent=(0,offset-onset), cmap='magma', clim=spec_clim);
    plt.axis('off')
    plt.xlim(0, offset-onset)


    plt.subplot(412)
    plt.plot(np.arange(onset_audio, offset_audio), audio, 'k')
    plt.axis('off')
    plt.xlim(onset_audio, offset_audio)
    sns.despine(bottom=True, right=True);

    plt.subplot(413)

    plt.hist(np.hstack(spikes_trunc), range=(0, (offset-onset)*sr_phys),
                 bins=int(len(audio)/int(hist_binsize*sr_audio)), histtype='step', color='k')
    plt.xticks([])
    plt.ylabel('counts \n ({} s bin)'.format(hist_binsize), rotation=0, labelpad=40, fontsize=14)
    plt.xlim(0, (offset-onset)*sr_phys)
    sns.despine(bottom=True, right=True);

    plt.subplot(414)
    for i in range(len(spikes_trunc)):
        plt.plot(spikes_trunc[i], [i]*len(spikes_trunc[i]), '|k')

    plt.xticks(np.arange(0, (offset-onset)*sr_phys, int(sr_phys*.5)),
               np.arange(0, (offset-onset)*sr_phys, int(sr_phys*.5))/sr_phys)
    plt.xlabel('time (s)')
    plt.ylabel('channel', rotation=0, labelpad=40, fontsize=14)
    plt.xlim(0, (offset-onset)*sr_phys)
    sns.despine(offset=10, left=False, right=True)
    plt.tight_layout()


def psth_channel(spikes, audio, onset_s_array, ephys_trigger,
                 pad = 1, hist_binsize=0.05, sr_audio=125000,
                 sr_phys=12500, n_channels=64, save_fig=False,
                 outname='', hide_plot=False, savedir=''):



    all_spikes = []
    #all_audio = []

    for onset in onset_s_array:

        dur = len(audio)/sr_audio
        offset = onset + dur + pad
        onset = onset - pad
        total_dur = dur + (2*pad)
        #onset_audio, offset_audio = int(onset*sr_audio)-ephys_trigger, int(offset*sr_audio)-ephys_trigger
        #audio = data_audio[0, onset_audio:offset_audio]
        #all_audio.append(audio)


        spikes_trunc = truncate_spikes(spikes,
                            onset = onset,
                            offset = offset,
                            ephys_trigger=ephys_trigger)

        all_spikes.append(spikes_trunc)

    #all_audio_mean = np.mean(np.array(all_audio), axis=0)
    psth_traces = []

    # all_spikes: list of n_onsets lists of 64 ndarrays of dim (n_spikes)
    # basically (n_onsets, 64, n_spikes)
    # want to transpose to (64, n_onsets, n_spikes)
    reshaped_data = list()
    for i in range(n_channels):
        element = [onset_trunc_spikes[i] for onset_trunc_spikes in all_spikes]  # len 'n_onsets' list of ndarrays (n_spikes,)
        reshaped_data.append(element)

    for i in range(n_channels):
        # For clarification, ch is staggered with dimensions (n_onsets, n_channels, n_spikes)
        # All spikes is a python list of len n_onsets, containing staggered ndarrays of dim (n_channels, n_spikes)
        # ch = np.vstack(all_spikes)[:,i]  # I believe this returns a matrix of spike locations for a single channel across all presentations
        ch = reshaped_data[i]

        plt.figure(figsize=(12,10))

        plt.subplot(411)
        plt.specgram(audio, NFFT=512, noverlap=256, Fs=sr_audio,
                     xextent=(pad,total_dur-pad), cmap='magma');
        plt.axis('off')
        plt.xlim(0, total_dur)


        plt.subplot(412)
        plt.plot(np.arange(pad*sr_audio, (pad+dur)*sr_audio), audio, 'k')
        plt.axis('off')
        plt.xlim(0, total_dur*sr_audio)
        sns.despine(bottom=True, right=True);


        plt.subplot(413)

        n, bins, patches = plt.hist(np.hstack(ch), range=(0, total_dur*sr_phys),
                                     bins=int(len(audio)/int(hist_binsize*sr_audio)),
                                    histtype='step', color='k')
        psth_traces.append(np.array([n, bins], dtype=object))

        plt.xticks([])
        plt.ylabel('counts \n ({} s bin)'.format(hist_binsize), rotation=0, labelpad=40, fontsize=14)
        plt.xlim(0, total_dur*sr_phys)
        sns.despine(bottom=True, right=True);


        plt.subplot(414)
        for j in range(len(ch)):
                plt.plot(ch[j], [j]*len(ch[j]), '|k')

        plt.xticks(np.arange(0, total_dur*sr_phys, int(sr_phys*.5)),
                    np.arange(0, total_dur*sr_phys, int(sr_phys*.5))/sr_phys)

        plt.xlabel('time (s)')
        plt.ylabel('sound trial', rotation=0, labelpad=40, fontsize=14)
        plt.xlim(0, total_dur*sr_phys)

        sns.despine()
        plt.title('Channel {}'.format(i+1))

        if save_fig == True:
            plt.savefig(os.path.join(savedir, 'channel{}{}.png'.format(i+1, outname)), dpi=300, transparent=False)

        if hide_plot == True:
            plt.close()

    np.save(os.path.join(savedir, 'psth_traces{}.npy'.format(outname)), np.array(psth_traces))


def rms(signal):
    rms = np.sqrt(np.mean(signal**2))
    return rms

def get_onsets(time_of_first_stim, time_between_stim, n_stim, stim_dur):

    """
    Get the onsets (in seconds) for all sound presentations in an experiment, given the first onset time.
    Note: this function is applicable to data acquired from 9/10/2021 to current (09/27/2021)

    Parameters:
    time_of_first_stim: float
        Time of the first audio presentation in seconds.
    time_between_stim: float
        This is the time in between the offset of the previous audio stim and the onset of the next audio stim.
    n_stim: int
        The number of stimulus presentations
    stim_dur: int or float
        The duration of the stimuli.

    Returns:
    onsets: 1-D array
        (n_stim, ) array on onsets (in seconds) for each audio stimulus.

    """

    # NOTE: times in SECONDS
    onsets =  np.arange(time_of_first_stim, (time_between_stim + stim_dur)*n_stim, (time_between_stim + stim_dur))

    #TODO: get time_between_stim, n_stim, and stim_dur from audio config file.

    return onsets


def load_data_new(exp_dir, phys_bandpass=(200, 2000), combine_audio=True):

    """
        Parameters:
        exp_dir: str
            directory where phys and analog data live
        phys_bandpass: tuple or boolean
            range (in Hz) to bandpass phys data. If False, raw data returned.
    """

    #search the exp directory for the physiology and analog data
    wm_file = glob(os.path.join(exp_dir, '*.bin'))[0]
    analog_file = glob(os.path.join(exp_dir, '*.h5'))[0]

    #load white matter data and check for bandpass argument...may take a while
    if phys_bandpass == False:
        print('Loading physiology data...')
        sr_phys, data_phys = load_wm(wm_file)
        print('No bandpass filtering.')
        print('Computing mean signal across all channels...')
        data_phys_mean = np.mean(data_phys, axis=0)
        print('Subtracting mean signal across all channels...')
        data_phys = data_phys - data_phys_mean.reshape((1, -1))
        print()

    else:
        print('Loading physiology data...')
        sr_phys, data_phys = load_wm(wm_file)
        print('Bandpassing from {}-{} Hz...'.format(phys_bandpass[0], phys_bandpass[1]))
        data_phys_filt = butter_bandpass_filter(data_phys, phys_bandpass[0], phys_bandpass[1], sr_phys)
        print('Computing mean signal across all channels...')
        data_phys_mean = np.mean(data_phys_filt, axis=0)
        print('Subtracting mean signal across all channels...')
        data_phys = data_phys_filt - data_phys_mean.reshape((1, -1))
        print()

    #load and truncate audio data
    print('Truncating audio...')
    data_audio, ephys_trigger_rising_edge = truncate_audio_new(analog_file, data_phys, combine_audio = combine_audio)
    print()

    return sr_phys, data_phys, data_audio, ephys_trigger_rising_edge


def truncate_audio_new(analog_file, data_phys, combine_audio = False):

    """
    Truncate beginning and end of audio to be aligned with physiology data and same length (in seconds)

    Parameters:
    exp_dir: str
        Directory where phys and analog data live

    data_phys: 2-D array
        output from load_wm() - a 2-D array that is (n_samples x n_channels)

    combine_audio: boolean
        if True, function ouputs a single 1-D audio array with signal average from all mics.
        if False, function outputs N 1-D audio arrays for each mic.

    Returns:
    audio: N-D array

    if combine_audio True:
        audio: 1-D array:
            average audio signal

    if combine_audio False:
        mic1, mic2: 2-D array:
            audio signals

    """

    #load analog file
    data_analog = h5py.File(analog_file, 'r')

    # load the ephys trigger sample number (this is where we will truncate the data at the beginning)
    ephys_trigger = data_analog['ephys_trigger'][0]
    #TODO return the rising trigger edge
    print('Ephys trigger detected at analog sample number {}'.format(ephys_trigger))
    print()

    #calculate the end truncation sample number
    #TODO: don't hard code the analog/ephys sampling ratio
    sampling_rate_ratio = 10
    end_trunc = (len(data_phys)*sampling_rate_ratio)

    _mic1 = data_analog['ai_channels']['ai0']

    mic1 =  _mic1[ephys_trigger:ephys_trigger+end_trunc]

    _mic2 = data_analog['ai_channels']['ai1']
    mic2 = _mic2[ephys_trigger:ephys_trigger+end_trunc]

    data_analog.close()

    if combine_audio == True:
        audio = 0.5 * (mic1 + mic2)
        return np.array([audio]), ephys_trigger

    else:
        return np.stack((mic1, mic2), axis=0), ephys_trigger


def h5_to_wav(dirname,  sr_audio=125000, mic_number=1):
    fns = glob(os.path.join(dirname, '*.h5'))

    for i in range(len(fns)):
        with h5py.File(fns[i], 'r') as f:
            audio = f['ai_channels']['ai{}'.format(mic_number-1)][()]
        outfile = os.path.join(dirname, os.path.split(fns[i])[-1][:-3] + '_mic{}.wav'.format(mic_number))
        wavfile.write(outfile, sr_audio, audio)
        print('Wrote data to: {}'.format(outfile))


