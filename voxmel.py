import os

import numpy as np
import numpy.ma as ma
from scipy.io.wavfile import write
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

from parselmouth import Sound
from tones import mixer
from librosa.core import hz_to_midi, midi_to_hz
from essentia.standard import MetadataReader, EqloudLoader



def extract_melody(fp=None, audio=None, sf=44100, quantise=False, retune=True,
                   pitch_method='praat', return_data=False, verbose=False, **kwargs):

    '''Automatically extract the musical note sequence from raw audio imput
    of natural speech.

    Parameters
    ----------
    fp : str, optional
        File path to audio file. If None then must supply `audio` (and associated
        `sf`). Default None.
    audio : array_like, optional
        Numpy array containing the audio sampled at sample frequency `sf`. Required
        if `fp` is not provided. Defualt None.
    sf : {32000, 44100, 48000}, optional
        The sample frequency (in Hertz) of the audio array `audio`, if provided.
        Ignored if `fp` is used. Default 44100.
    quantise : bool, optional
        Whether to quantise the note values to match those of MIDI notes (where A=440Hz).
        Default False.
    retune : bool, optional
        Whether to retune the pitches such that on average pitches are close to MIDI notes.
        Default False.
    pitch_method : {'praat', 'crepe'}, optional
        Algorithm to extract the pitch contour. Defualt 'praat'.
    return_data : bool, optional
        Return the pitch and intensity contour as well. Default False.
    verbose : bool, optional
        Print details. Default False.
    **kwargs : optional
        Keyword arguments to pass onto `segment_notes()` and `get_notes()`.


    Returns
    -------
    ns : array
        Note start times (in seconds).
    nl : array
        Note lengths (in seconds).
    nv : array
        Note values (in Hertz).
    data : dict, optional
        Containing `pitch`, `intensity` data. Returned only if `return_data` is True.


    Raises
    ------
    ValueError
        If neither `fp` nor 'audio' is provided.
    ValueError
        If `fp` cannot be found.
    ValueError
        If `sf` not in {32000, 44100, 48000}.
    '''

    if verbose:
        print('== Starting extracting notes with kwargs', kwargs)

    if fp is not None:
        if not os.path.isfile(fp):
            raise ValueError('Path {} not found.'.format(fp))

        sf = MetadataReader(filename=fp)()[10]
        audio = EqloudLoader(filename=fp, sampleRate=sf)()

    elif audio is None:
        raise ValueError('Must provide either filepath (`path`) or array of audio data (`audio`).')

    if sf not in {32000, 44100, 48000}:
        raise ValueError('Sample frequency `sf` {} not in {32000, 44100, 48000}.'.format(sf))

    # Sound object as parsed by Praat
    sound = Sound(values=np.asarray(audio, dtype=np.float64), sampling_frequency=sf)
    end_time = sound.end_time

    if verbose:
        print('- Loaded audio: sf={}, length={:.2f} seconds'.format(sf, end_time))


    # Pitch and Intensity vectors

    if pitch_method == 'praat':
        ts = np.arange(0, end_time, 0.01)
        pitch = sound.to_pitch_ac(time_step=0.01, octave_jump_cost=0.6)
        p = []

        for t in ts:
            p.append(pitch.get_value_at_time(t))

    elif pitch_method == 'crepe':
        import crepe
        # check if f0.csv exists for file
        if fp is not None:
            f0_path = fp[:-3] + 'f0.csv'
            if not os.path.exists(f0_path):
                crepe.process_file(fp)

            data = np.genfromtxt(f0_path, dtype=None, skip_header=1, delimiter=',')

        else:
            data = crepe.predict(audio, sf, verbose=0)
            data = np.stack(data[:3]).T

        ts = data[:, 0]
        p = data[:, 1]
        p[np.where(data[:, 2] < 0.5)] = np.nan

    else:
        raise ValueError('`pitch_method` must be in {"praat", "crepe"}')


    intensity = sound.to_intensity(minimum_pitch=50, time_step=0.01)
    I = []

    for t in ts:
        I.append(intensity.get_value(t))

    p = np.array(p)
    p[p == 0] = np.nan
    I = np.array(I)
    I[np.isnan(I)] = 0

    if verbose:
    	print('- Computed pitch and intensity contours.')

    if retune:
        if verbose:
            print('- Retuning pitches.')
        p = retunePitches(p)


    _, _, nuclei = segment_notes(I, p, ts, verbose=verbose, **kwargs)
    ns, nl, nv = get_notes(I, p, ts, nuclei, verbose=verbose, **kwargs)

    if verbose:
        print('== Found {} notes.'.format(len(nv)))

    if quantise:
        if verbose:
            print('- quantising note values')
        nv = quantise_notes(nv)

    if verbose:
        print('\n== Done\n')

    if return_data:
        data = {'pitch': p, 'intensity': I, 'ts': ts, 'nuclei': nuclei}
        return ns, nl, nv, data
    else:
        return ns, nl, nv



def segment_notes(intensity, pitch, ts, maxDip=0.125, minDipBefore=2.0, minDipAfter=0.0,
                  threshold=0, verbose=False, **kwargs):

    '''Identifies the note boundaries from the nuclei of the vocalic sections of the audio.

    Parameters
    ----------
    intensity : array-like
        The intensity vector of the audio.
    pitch : array-like
        The pitch vector containing the F0 frequency contour. Required to filter unpitched
        nuclei.
    ts : array-like
        The timesteps of pitch and intensity.
    maxDip : float, optional
        Default 0.5.
    minDipBefore : float, optional
        Default 2.1.
    minDipAfter : float, optional
        Default 0.5.
    threshold : float, optional
        Default 0.0.
    verbose : bool, optional
        Print details. Default False.

    Returns
    -------
    peaks_init : array
        Time stamps of all the peaks of I (includes time 0).
    peaks : array
        Time stamps of filtered peaks of I.
    nuclei : array
        Time stamps of nuclei after final round of peak filtering.

    '''

    if verbose:
        print('== Segmenting notes.')
        print(f'intensity.shape={intensity.shape}, pitch.shape={pitch.shape}, ts.shape={ts.shape}, \
        maxDip={maxDip}, minDipBefore={minDipBefore}, minDipAfter={minDipAfter}, threshold={threshold}')
        print(f'{kwargs}')

    peaks_init = np.array([0] + list(ts[find_peaks(intensity)[0]]))

    if verbose:
        print('Found', len(peaks_init), 'peaks')

    if verbose:
        print('- First round of filtering:')
    remove = []
    for i in range(len(peaks_init)):
        p_i = peaks_init[i]
        try:
            p_i1 = peaks_init[i+1]
        except:
            p_i1 = ts[-1]

        margin = 0.1
        pitch_around = pitch[np.where(np.logical_and(ts>=p_i-margin, ts<p_i+margin))]

        #peak_idx = np.argmin(np.abs(ts - p_i))
        peak_idx = np.searchsorted(ts, p_i)

        if peak_idx > 0 and peak_idx < len(intensity) - 2:
            I_p = np.mean(intensity[peak_idx-1:peak_idx+1])
        else:
            I_p = intensity[0]

        if I_p < threshold:
        #if intensity.get_value(time=p_i) < threshold:
            # Filter 1: below threshold...
            remove.append(i)
            if verbose: print(' - peak at {:.4f} too low'.format(p_i))

        elif any(np.isfinite(pitch_around)):
            # Filter 2: voiced/pitched...
            i_range = intensity[np.where(np.logical_and(ts>=p_i, ts<p_i1))]

            if np.abs(intensity[peak_idx] - np.min(i_range)) < maxDip:
                # Filter 3: small dip...
                remove.append(i)
                if verbose: print(' - peak at {:.4f} too small'.format(p_i))
        else:
            remove.append(i)
            if verbose: print(' - peak at {:.4f} unpitched'.format(p_i))

    if verbose:
        print('- (Removing', len(remove), 'peaks)')
    peaks = np.array([peaks_init[i] for i in range(len(peaks_init)) if i not in remove])

    if verbose:
        print('- Second round of filtering:')
    nuclei = []
    for i in range(0, len(peaks)):
        if i == 0:
            I_before = 0
        else:
            I_before = np.min(intensity[np.where(np.logical_and(ts>=peaks[i-1], ts<peaks[i]))])

        if i < len(peaks) - 1:
            I_after = np.min(intensity[np.where(np.logical_and(ts>=peaks[i], ts<peaks[i+1]))])
        else:
            I_after = 0

        #I_peak = intensity[np.argmin(np.abs(ts - peaks[i]))]

        peak_idx = np.searchsorted(ts, peaks[i])

        if peak_idx > 0:
            I_p = np.mean(intensity[peak_idx-1:peak_idx+1])
        else:
            I_p = intensity[0]

        a = np.abs(I_p - I_after) >= minDipAfter
        b = np.abs(I_p - I_before) >= minDipBefore

        if a and b:
            nuclei.append(peaks[i])
        else:
            if verbose: print(' - peak at {:.4f} too small'.format(peaks[i]))
    if verbose:
        print('{} nuclei identified.'.format(len(nuclei)))

    return peaks_init, peaks, np.array(nuclei)



def get_notes(intensity, pitch, ts, nuclei, minLength=0.0, unstable=5.0, maxUnstable=13.5,
              method='derivatives', verbose=False, **kwargs):

    '''Returns the note value of the segmented note regions.

    Parameters
    ----------
    intensity : array-like
        The intensity vector of the audio.
    pitch : array-like
        The pitch vector containing the F0 frequency contour.
    ts : array-like
        The timesteps of pitch and intensity.
    nuclei : array-like
        List of timestamps of identified nuclei.
    minLength : float, optional
        Default 0.05.
    unstable : float, optional
        Default 4.5.
    maxUnstable : float, optional
        Default 8.8.
    method : {'derivatives', 'mean', 'mean_stable', 'max', 'max_stable'}
        Default 'derivatives'.
    verbose : bool, optional
        Print details. Default False.

    Returns
    -------
    notes : tuple
        Tuple of note start times, note lengths and note values.

    '''

    if verbose:
        print('\n== Extracting melody using', method, 'method')
    #if method not in ('derivatives', 'mean', 'mean_stable', 'max', 'max_stable'):
    #    raise ValueError('method {} not in ("derivatives", "mean", "mean_stable", "max", "max_stable")'.format(method))

    #I = intensity.values[0]
    #t = intensity.ts()
    #dt = intensity.dt

    #I_t = np.linspace(0, end_time, len(intensity))
    dt = ts[1] - ts[0]

    #t_p = np.linspace(0, end_time, len(pitch))

    if verbose:
        print('Cleaning pitch track')
    pitch = clean_f0(pitch)
    #f0 = pitch.selected_array['frequency']
    #f0 = clean_f0(f0)

    #t_p = pitch.ts()
    note_start, note_length, note_pitch = [], [], []

    # include first boundary from first voiced pitch
    try:
        note_boundaries = [ts[np.where(np.isfinite(pitch))][0]]
    except IndexError:
        # f0 all np.nan, return empty
        if verbose:
            print('No note boundaries found, empty melody.')
        return np.array([]), np.array([]), np.array([])

    for n1, n2 in zip(nuclei[:-1], nuclei[1:]):
        I_between = intensity[np.where(np.logical_and(ts>=n1, ts<n2))]
        if len(I_between) > 0:
            note_boundaries.append(n1 + dt*np.argmin(I_between))

    # include last time point (when I finally dips below 50)
    try:
        note_boundaries.append(ts[np.where(intensity > 50)[0][-1]])
    except IndexError:
        note_boundaries.append(ts[-1])

    for t1, t2 in zip(note_boundaries[:-1], note_boundaries[1:]):
        p_note = pitch[np.where(np.logical_and(t1 <= ts, ts < t2))]
        if len(p_note) == 0 or sum(np.isfinite(p_note))/len(p_note) < 0.25:
            if verbose: print(' - note at {:.2f}: too unpitched, ignoring'.format(t1))
            continue

        try:
            start = max(t1, t1 + dt*np.argwhere(np.isfinite(p_note))[0][0])
        except IndexError:
            if verbose: print(' - note at {:.2f}: start not found, ignoring'.format(t1))
            continue

        if t2 - start < minLength:
            # note too short...
            if verbose: print(' - note at {:.2f}: too short ({} < {}), ignoring'.format(t1, t2-start, minLength))
            continue

        p_note = p_note[np.where(np.isfinite(p_note))]
        dp = np.diff(p_note, n=1)
        if len(dp) == 0 or len(p_note) < 3:
            continue

        note_start.append(start)
        note_length.append(t2 - start)

        if method in ('derivatives', 'max', 'max_stable'):
            #prop_unstable = np.sum(np.abs(dp) > unstable)/len(dp)
            p_stability = np.mean(np.abs(dp))
            #p_range = np.log2(np.max(p_note)) - np.log2(np.min(p_note))

            if p_stability > maxUnstable:
                # unstable, take max if increasing, min otherwise...
                if verbose: print(' - note at {:.2f}: unstable ({:.2f} > {:.2f})'.format(t1, p_stability, maxUnstable))
                mid = len(p_note) // 2
                note_pitch.append(np.nanmean(p_note[mid:]))

            else:
                # mostly stable, take mean of last largest stable region
                if verbose: print(' - note at {:.2f}: stable ({:.2f} <= {:.2f})'.format(t1, p_stability, maxUnstable))
                regions = np.ma.clump_unmasked(np.ma.masked_greater(np.abs(dp), unstable))
                if verbose:
                    print('\t ({} stable regions.)'.format(len(regions)))

                if len(regions) > 0:
                    #longest_region = list(reversed(regions))[0]
                    region_lengths = [s.stop - s.start for s in reversed(regions)]
                    longest_region = list(reversed(regions))[np.argmax(region_lengths)]
                    if method in ('max', 'max_stable'):
                        note_pitch.append(np.nanmax(p_note[longest_region]))
                    else:
                        note_pitch.append(np.nanmean(p_note[longest_region]))

                else:
                    if method in ('max', 'max_stable'):
                        note_pitch.append(np.nanmax(p_note))
                    else:
                        note_pitch.append(np.nanmean(p_note))

        elif method == 'mean':
            note_pitch.append(np.nanmean(p_note))

        elif method == 'mean_stable':
            # find most stable regions
            p_stable = p_note[np.where(dp < unstable)]
            if len(p_stable) < 2:
                # fall back to method mean for very unstable notes
                note_pitch.append(np.nanmean(p_note))
            else:
                note_pitch.append(np.nanmean(p_stable))
        else:
            raise(ValueError('method {} not in ("derivatives", "mean", "mean_stable", "max", "max_stable")'.format(method)))

    return np.array(note_start), np.array(note_length), np.array(note_pitch)


def retunePitches(pitch, tuning=440):
    '''Transposes pitches such that on average frequencies are close to 12 tone equal temperment
    tuning at reference with respect to a reference pitch.

    Parameters
    ----------
    pitch : array-like
        The pitch contour (in Hertz)
    tuning : float (optional)
        Frequency of reference pitch A4 tune to. Default 440.

    Returns
    -------
    retuned : array
        Retuned pitches.

    '''

    # mask invalid pitchs (i.e. NaNs and 0)
    pitch = ma.masked_invalid(pitch)
    pitch = ma.masked_equal(pitch, 0)

    invalid_idxs = pitch.mask

    # convert to semitones values, such that 1 step corresponds to a semitone
    n = 12 * np.log2(pitch / tuning)

    # take the modulo 1 of each pitch
    mod = np.mod(n, 1)

    # convert into an 'angle' around a circle
    alpha = np.pi * 2 * mod

    # compute circular mean
    mean_sin = np.mean(np.sin(alpha))
    mean_cos = np.mean(np.cos(alpha))
    mean = np.arctan2(mean_sin, mean_cos)

    # average distance pitches are away from notes
    offset = mean / (np.pi * 2)
    n += offset
    retuned = tuning * np.power(2, n / 12)

    retuned = np.array(retuned)
    retuned[invalid_idxs] = np.nan

    return retuned



def quantise_notes(notes):
    '''Quantises an array of note values in Hertz to those in a 12 tone equal temperment tuning.

    Parameters
    ----------
    notes : array-like or tuple
        Either a list of note values, or a tuple of (note start times, note lengths, note values).


    Returns
    -------
    notes : array
        List of quantised notes, matching the shape of `notes`.


    Raises
    ------
    ValueError
        If `notes` is not in a valid format.

    '''

    if len(np.shape(notes)) == 2:
        notes = np.asarray(notes)
        notes[2] = midi_to_hz(np.round(hz_to_midi(notes[2])))
        return notes
    elif len(np.shape(notes)) == 1:
        return midi_to_hz(np.round(hz_to_midi(notes)))
    else:
        raise ValueError('`notes` must be one or two dimenstional array-like lists.')






def clean_f0(f0):
    '''Removes outlier points and spikes.

    Parameters
    ----------
    f0 : array-like
        The pitch contour to be cleaned.

    Returns
    -------
    clean : array
        The cleaned pitch contour.
    '''

    clean = np.copy(f0)
    # isolate each continuous segment...
    clean[clean==0] = np.nan
    segments = np.ma.clump_unmasked(np.ma.masked_invalid(clean))

    for s in segments:
        dp = np.diff(np.log2(clean[s]), n=1, prepend=0)
        spikes, _ = find_peaks(np.abs(dp), height=0.4)

        # remove points between very close spikes (or end of segments)...
        idxs = [s.start] + [s.start+i for i in spikes] + [s.stop]
        for j in range(len(idxs)-1):
            i1, i2 = idxs[j], idxs[j+1]
            if i2 - i1 < 2:
                clean[i1:i2] = np.nan

    return clean


def plot_notes(notes, ax=None, octave=0, thickness=0.2, **kwargs):
    '''Plots the note sequence for visualisation.

    Parameters
    ----------
    notes : tuple
        Tuple of note start times, note lengths, note values.
    ax : matplotlib.Axes, optional
        If provided, plot notes on custom axis. Default None.
    octave : float, optional
        How many octaves to shift the notes by, can be non-integer. Default 0.
    thickness : float, optional
        Thickness of note lines. Default 3.
    **kwargs : optional
        Keyword arguments to pass to `ax.fill`, (which draws the notes).

    Returns
    -------
    ax : matplotlib.Axes
        Axes object
    '''


    if not ax:
        fig, ax = plt.subplots()
        ax.set_xlabel('time')
        ax.set_ylabel('MIDI note')

    h = thickness
    for ns, nl, nv in zip(*notes):
        freq = hz_to_midi(nv) + 12 * octave
        lines = ax.fill([ns, ns+nl-0.02, ns+nl-0.02, ns], [freq-h, freq-h, freq+h, freq+h], **kwargs)

    return ax


def midi_to_notes(fp, instrument=0, verbose=False):
    '''Converts a MIDI file into a sequence of notes in the same format as the
    output of `get_notes`. Useful for comparing the extracted melody with a
    MIDI transcription.

    Parameters
    ----------
    fp : string or file
        File path or file handle to MIDI file
    instrument : int, optional
        Index of the instrument (track) to convert. Default 0.
    verbose: bool
        Print process of reading MIDI file. Default False.

    Returns
    -------
    notes : tuple
        Tuple of note start times, note lengths, note values.
    '''

    import pretty_midi as pm

    if verbose:
        print('\n- Loading {}'.format(fp))

    mid = pm.PrettyMIDI(fp)
    if verbose:
        print('-- {} instruments, reading {}'.format(len(mid.instruments), instrument))
    ins = mid.instruments[instrument]

    ns = []
    nl = []
    nv = []

    if verbose:
        print('-- {} notes'.format(len(ins.notes)))

    for n in ins.notes:
        ns.append(n.start)
        nl.append(n.end - n.start)
        nv.append(midi_to_hz(n.pitch))
        if verbose:
            print('  ({:.2f}, {:.2f}, {:.2f})'.format(n.start, n.end-n.start, midi_to_hz(n.pitch)))

    return ns, nl, nv


def create_audio(notes, end_time=None, save_path=None, sf=44100, octave=0, **kwargs):
    '''Creates an audiofile from a collection of notes.

    Parameters
    ----------
    notes : tuple
        Tuple of note start times, note lengths, note values.
    end_time : float, optional
        The full length of the final stimulus. If not provided, end when last note has
        finished. Default None.
    save_path : string, optional
        If provided, save the audio to `save_path`. Default None.
    sf : int, optional
        Sample frequency for the generated audio. Default 44100.
    octave : float, optional
        How many octaves to shift the notes by, can be non-integer. Default 0.
    kwargs : optional
        Keyword arguments to pass onto tones.Track (see https://tones.readthedocs.io/en/latest/tones.html#tones.mixer.Track)


    Returns
    -------
    samples : array
        Audio waveform.
    '''


    mix = mixer.Mixer(sample_rate=sf)
    if 'attack' not in kwargs:
        kwargs['attack'] = 0.08
    if 'decay' not in kwargs:
        kwargs['decay'] = 0.08

    mix.create_track(0, **kwargs)

    t = 0
    for ns, nl, nv in zip(*notes):
        if ns > t:
            # add silence between notes
            sl = ns - t
            mix.add_silence(0, sl)

        mix.add_tone(0, frequency=nv*(2**octave), duration=nl, amplitude=0.8)
        t = ns + nl

    # add silence at end...
    if end_time and end_time > t:
        sl = end_time - t
        mix.add_silence(0, sl)

    samples = mix.mix()

    if save_path:
        mix.write(path)

    return samples



