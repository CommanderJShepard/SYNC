import librosa
import numpy as np

def analyze_audio(audio_data, sampling_rate, analysis_options):
    """
    Analyzes the audio data based on the provided options.

    Args:
        audio_data (np.ndarray): The audio time series.
        sampling_rate (int): The sampling rate of the audio.
        analysis_options (dict): A dictionary of boolean flags for different analyses.

    Returns:
        dict: A dictionary containing the results of the analysis.
    """
    analysis_results = {}
    y_harmonic, y_percussive = None, None

    # HPSS is needed for multiple features, so run it once if any are selected.
    if analysis_options.get("Percussive Elements") or analysis_options.get("Harmonic Waveform") or analysis_options.get("Percussive Waveform"):
        y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
        if analysis_options.get("Harmonic Waveform"):
            analysis_results['harmonic_waveform'] = y_harmonic
        if analysis_options.get("Percussive Waveform"):
            analysis_results['percussive_waveform'] = y_percussive

    # BPM and Beats
    if analysis_options.get("BPM & Beats"):
        tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sampling_rate)
        beat_times = librosa.frames_to_time(beat_frames, sr=sampling_rate)
        analysis_results['beats'] = beat_times.tolist()
        analysis_results['estimated_bpm'] = tempo.item()

    # Onsets
    if analysis_options.get("Onsets (Transients)"):
        onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sampling_rate)
        onset_times = librosa.frames_to_time(onset_frames, sr=sampling_rate)
        analysis_results['onsets'] = onset_times.tolist()

    # Percussive Onsets
    if analysis_options.get("Percussive Elements"):
        if y_percussive is not None:
            percussive_onset_frames = librosa.onset.onset_detect(y=y_percussive, sr=sampling_rate)
            percussive_onset_times = librosa.frames_to_time(percussive_onset_frames, sr=sampling_rate)
            analysis_results['percussive_onsets'] = percussive_onset_times.tolist()

    # RMS Energy
    if analysis_options.get("Energy (RMS)"):
        rms = librosa.feature.rms(y=audio_data)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sampling_rate)
        analysis_results['rms_energy'] = list(zip(times.tolist(), rms.tolist()))

    # Chromagram
    if analysis_options.get("Chromagram"):
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sampling_rate)
        analysis_results['chroma'] = chroma

    # Tempogram
    if analysis_options.get("Tempogram"):
        # Use a subset of the audio for tempogram if it's too long, to avoid performance issues
        duration = librosa.get_duration(y=audio_data, sr=sampling_rate)
        y_for_tempo = audio_data
        if duration > 300: # limit to 5 minutes for tempogram
            y_for_tempo = audio_data[:sampling_rate * 300]

        onset_env = librosa.onset.onset_detect(y=y_for_tempo, sr=sampling_rate, units='time')
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sampling_rate)
        analysis_results['tempogram'] = tempogram
        analysis_results['tempo_times'] = librosa.times_like(tempogram)


    return analysis_results
