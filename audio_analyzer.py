import librosa
import numpy as np

def analyze_audio(audio_data, sampling_rate, analysis_options):
    """
    Analyzes the audio data based on the provided options.

    Args:
        audio_data (np.ndarray): The audio time series.
        sampling_rate (int): The sampling rate of the audio.
        analysis_options (dict): A dictionary of boolean flags for different analyses.
            Expected keys: "BPM & Beats", "Onsets (Transients)", "Energy (RMS)", "Percussive Elements".

    Returns:
        dict: A dictionary containing the results of the analysis.
    """
    analysis_results = {}

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

    # RMS Energy
    if analysis_options.get("Energy (RMS)"):
        rms = librosa.feature.rms(y=audio_data)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sampling_rate)
        analysis_results['rms_energy'] = list(zip(times.tolist(), rms.tolist()))

    # Percussive Elements
    if analysis_options.get("Percussive Elements"):
        y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
        percussive_onset_frames = librosa.onset.onset_detect(y=y_percussive, sr=sampling_rate)
        percussive_onset_times = librosa.frames_to_time(percussive_onset_frames, sr=sampling_rate)
        analysis_results['percussive_onsets'] = percussive_onset_times.tolist()

    return analysis_results
