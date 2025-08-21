import unittest
import librosa
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the audio_analyzer module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from audio_analyzer import analyze_audio

class TestAudioAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load a sample audio file once for all tests."""
        # Using a built-in example audio file from librosa
        cls.audio_path = librosa.example('trumpet')
        cls.y, cls.sr = librosa.load(cls.audio_path)

    def test_analyze_audio_all_options(self):
        """
        Test the analyze_audio function with all analysis options enabled.
        """
        analysis_options = {
            "BPM & Beats": True,
            "Onsets (Transients)": True,
            "Energy (RMS)": True,
            "Percussive Elements": True,
            "Harmonic Waveform": True,
            "Percussive Waveform": True,
            "Chromagram": True,
            "Tempogram": True,
        }

        results = analyze_audio(self.y, self.sr, analysis_options)

        # Check that all expected keys are in the results
        self.assertIn('estimated_bpm', results)
        self.assertIn('beats', results)
        self.assertIn('onsets', results)
        self.assertIn('rms_energy', results)
        self.assertIn('percussive_onsets', results)
        self.assertIn('harmonic_waveform', results)
        self.assertIn('percussive_waveform', results)
        self.assertIn('chroma', results)
        self.assertIn('tempogram', results)

        # Check the types and shapes of the results
        self.assertIsInstance(results['estimated_bpm'], float)
        self.assertIsInstance(results['beats'], list)
        self.assertIsInstance(results['harmonic_waveform'], np.ndarray)
        self.assertIsInstance(results['chroma'], np.ndarray)
        self.assertIsInstance(results['tempogram'], np.ndarray)
        self.assertEqual(results['harmonic_waveform'].shape, self.y.shape)
        self.assertEqual(results['chroma'].shape[0], 12) # 12 pitch classes
        self.assertTrue(results['tempogram'].shape[0] > 0)


    def test_analyze_audio_no_options(self):
        """
        Test the analyze_audio function with all analysis options disabled.
        """
        analysis_options = {} # Empty dict
        results = analyze_audio(self.y, self.sr, analysis_options)
        self.assertEqual(len(results), 0)

    def test_analyze_audio_hpss_dependency(self):
        """
        Test that HPSS is run when only percussive onsets are requested.
        """
        analysis_options = { "Percussive Elements": True }
        results = analyze_audio(self.y, self.sr, analysis_options)
        self.assertIn('percussive_onsets', results)
        self.assertNotIn('harmonic_waveform', results) # Should not be in results unless requested
        self.assertNotIn('percussive_waveform', results)

    def test_analyze_audio_chroma_only(self):
        """
        Test requesting only the chromagram.
        """
        analysis_options = { "Chromagram": True }
        results = analyze_audio(self.y, self.sr, analysis_options)
        self.assertIn('chroma', results)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results['chroma'], np.ndarray)


if __name__ == '__main__':
    unittest.main()
