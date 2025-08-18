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
            "Percussive Elements": True
        }

        results = analyze_audio(self.y, self.sr, analysis_options)

        # Check that all expected keys are in the results
        self.assertIn('estimated_bpm', results)
        self.assertIn('beats', results)
        self.assertIn('onsets', results)
        self.assertIn('rms_energy', results)
        self.assertIn('percussive_onsets', results)

        # Check the types of the results
        self.assertIsInstance(results['estimated_bpm'], float)
        self.assertIsInstance(results['beats'], list)
        self.assertIsInstance(results['onsets'], list)
        self.assertIsInstance(results['rms_energy'], list)
        self.assertIsInstance(results['percussive_onsets'], list)

        # Check that lists are not empty (for this particular sample)
        self.assertTrue(len(results['beats']) > 0)
        self.assertTrue(len(results['onsets']) > 0)
        self.assertTrue(len(results['rms_energy']) > 0)
        self.assertTrue(len(results['percussive_onsets']) > 0)

        # Check content of lists
        self.assertIsInstance(results['beats'][0], float)
        self.assertIsInstance(results['onsets'][0], float)
        self.assertIsInstance(results['rms_energy'][0], tuple)
        self.assertIsInstance(results['percussive_onsets'][0], float)

    def test_analyze_audio_no_options(self):
        """
        Test the analyze_audio function with all analysis options disabled.
        """
        analysis_options = {
            "BPM & Beats": False,
            "Onsets (Transients)": False,
            "Energy (RMS)": False,
            "Percussive Elements": False
        }

        results = analyze_audio(self.y, self.sr, analysis_options)

        # Check that the results dictionary is empty
        self.assertEqual(len(results), 0)

    def test_analyze_audio_some_options(self):
        """
        Test the analyze_audio function with only some options enabled.
        """
        analysis_options = {
            "BPM & Beats": True,
            "Onsets (Transients)": False,
            "Energy (RMS)": True,
            "Percussive Elements": False
        }

        results = analyze_audio(self.y, self.sr, analysis_options)

        # Check for expected keys
        self.assertIn('estimated_bpm', results)
        self.assertIn('beats', results)
        self.assertIn('rms_energy', results)

        # Check that other keys are not present
        self.assertNotIn('onsets', results)
        self.assertNotIn('percussive_onsets', results)

if __name__ == '__main__':
    unittest.main()
