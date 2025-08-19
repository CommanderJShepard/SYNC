import tkinter as tk
from tkinter import ttk, filedialog
import json
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pygame
from audio_analyzer import analyze_audio

class RhythmKickboxingLevelEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Rhythm Kickboxing Level Editor")
        self.root.geometry("1200x800")

        self.audio_file_path = None
        self.audio_data = None
        self.sampling_rate = None

        self.is_playing = False
        self.is_paused = False
        self.playback_position_marker = None

        self.choreography = []
        self.choreography_markers = []

        self.strike_types = [
            "Left Jab", "Right Cross", "Lead Hook", "Rear Hook",
            "Lead Uppercut", "Rear Uppercut", "Duck", "Slip Left", "Slip Right"
        ]

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # A. Controls Section
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        controls_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.setup_controls(controls_frame)

        # B. Visualization Section
        self.visualization_frame = ttk.LabelFrame(main_frame, text="Song Map", padding="10")
        self.visualization_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.visualization_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        # Initially hide the plot
        self.ax.set_visible(False)
        self.canvas.draw()

    def setup_controls(self, parent_frame):
        # ... (same as before)
        # File Operations Frame
        file_ops_frame = ttk.Frame(parent_frame)
        file_ops_frame.pack(fill=tk.X, pady=5)

        self.upload_btn = ttk.Button(file_ops_frame, text="Upload Song", command=self.upload_song)
        self.upload_btn.pack(side=tk.LEFT, padx=5)

        self.analyze_btn = ttk.Button(file_ops_frame, text="Analyze Music", command=self.analyze_music, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)

        self.export_btn = ttk.Button(file_ops_frame, text="Export Level", command=self.export_level, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=5)

        # Analysis Options Frame
        analysis_options_frame = ttk.LabelFrame(parent_frame, text="Analysis Options")
        analysis_options_frame.pack(fill=tk.X, pady=10, ipady=5)

        self.analysis_vars = {
            "BPM & Beats": tk.BooleanVar(value=True),
            "Onsets (Transients)": tk.BooleanVar(value=True),
            "Energy (RMS)": tk.BooleanVar(value=True),
            "Percussive Elements": tk.BooleanVar(value=True)
        }

        for i, (text, var) in enumerate(self.analysis_vars.items()):
            ttk.Checkbutton(analysis_options_frame, text=text, variable=var).pack(side=tk.LEFT, padx=10)

        # Playback Controls Frame
        playback_frame = ttk.Frame(parent_frame)
        playback_frame.pack(fill=tk.X, pady=5)

        self.play_pause_btn = ttk.Button(playback_frame, text="Play", command=self.toggle_playback, state=tk.DISABLED)
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(playback_frame, text="Stop", command=self.stop_audio, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

    def upload_song(self):
        self.audio_file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.mp3 *.wav *.ogg *.flac")]
        )
        if not self.audio_file_path:
            return

        try:
            # Stop any previous playback
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()

            self.audio_data, self.sampling_rate = librosa.load(self.audio_file_path, sr=None)

            # Initialize pygame mixer with the song's sampling rate
            pygame.mixer.quit()
            pygame.mixer.init(frequency=self.sampling_rate)
            pygame.mixer.music.load(self.audio_file_path)

            self.plot_waveform()
            self.analyze_btn['state'] = tk.NORMAL
            self.play_pause_btn['state'] = tk.NORMAL
            self.stop_btn['state'] = tk.NORMAL
            self.export_btn['state'] = tk.NORMAL
            self.is_playing = False
            self.is_paused = False
            self.play_pause_btn.config(text="Play")

        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load audio file: {e}")

    def plot_waveform(self):
        self.ax.clear()
        librosa.display.waveshow(self.audio_data, sr=self.sampling_rate, ax=self.ax, alpha=0.5)
        self.ax.set_title("Song Waveform")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_visible(True)
        self.canvas.draw()

    def analyze_music(self):
        # ... (same as before)
        print("Analyzing music...")
        self.plot_waveform()  # Re-plot to clear old analysis

        analysis_options = {key: var.get() for key, var in self.analysis_vars.items()}
        self.analysis_results = analyze_audio(self.audio_data, self.sampling_rate, analysis_options)

        # --- Visualization of Analysis Results ---
        if "beats" in self.analysis_results:
            beat_times = self.analysis_results['beats']
            bpm = self.analysis_results.get('estimated_bpm', 0)
            self.ax.vlines(beat_times, -1, 1, color='b', linestyle='--', label=f'Beats (BPM: {bpm:.2f})')
        if "onsets" in self.analysis_results:
            onset_times = self.analysis_results['onsets']
            self.ax.vlines(onset_times, -1, 1, color='r', linestyle=':', label='Onsets')
        if "rms_energy" in self.analysis_results:
            rms_data = self.analysis_results['rms_energy']
            times = [p[0] for p in rms_data]
            rms_values = np.array([p[1] for p in rms_data])
            if rms_values.size > 0:
                rms_normalized = rms_values / np.max(rms_values) if np.max(rms_values) > 0 else rms_values
                self.ax.plot(times, rms_normalized, color='g', alpha=0.6, label='Energy (RMS)')
        if "percussive_onsets" in self.analysis_results:
            percussive_onset_times = self.analysis_results['percussive_onsets']
            self.ax.vlines(percussive_onset_times, -1, 1, color='y', linestyle='-.', label='Percussive Onsets')

        self.ax.legend()
        self.canvas.draw()
        print("Analysis complete.")

    def toggle_playback(self):
        if not self.is_playing:
            # Start playing
            self.is_playing = True
            self.is_paused = False
            pygame.mixer.music.play()
            self.play_pause_btn.config(text="Pause")
            self.update_playback_marker()
        elif self.is_playing and not self.is_paused:
            # Pause
            self.is_paused = True
            pygame.mixer.music.pause()
            self.play_pause_btn.config(text="Resume")
        elif self.is_playing and self.is_paused:
            # Resume
            self.is_paused = False
            pygame.mixer.music.unpause()
            self.play_pause_btn.config(text="Pause")

    def stop_audio(self):
        pygame.mixer.music.stop()
        self.is_playing = False
        self.is_paused = False
        self.play_pause_btn.config(text="Play")
        if self.playback_position_marker:
            self.playback_position_marker.remove()
            self.playback_position_marker = None
            self.canvas.draw_idle()

    def update_playback_marker(self):
        if pygame.mixer.music.get_busy() and self.is_playing and not self.is_paused:
            # Get position in seconds
            current_time = pygame.mixer.music.get_pos() / 1000.0

            if self.playback_position_marker:
                self.playback_position_marker.remove()

            self.playback_position_marker = self.ax.axvline(current_time, color='purple', lw=2)
            self.canvas.draw_idle()

            # Schedule next update
            self.root.after(50, self.update_playback_marker)
        elif not pygame.mixer.music.get_busy():
            # Song finished
            self.stop_audio()

    def on_canvas_click(self, event):
        # ... (same as before)
        if event.inaxes != self.ax:
            return
        timestamp = event.xdata
        if event.button == 1:
            self.show_choreography_menu(event.x, event.y, timestamp)
        elif event.button == 3:
            self.delete_choreography_note(timestamp)

    def show_choreography_menu(self, x, y, timestamp):
        # ... (same as before)
        menu = tk.Menu(self.root, tearoff=0)
        for strike in self.strike_types:
            menu.add_command(label=strike, command=lambda s=strike: self.add_choreography_note(timestamp, s))
        try:
            menu.tk_popup(x, self.root.winfo_y() + y, 0)
        finally:
            menu.grab_release()

    def add_choreography_note(self, timestamp, strike_type):
        # ... (same as before)
        note = {"timestamp": timestamp, "type": "strike", "value": strike_type}
        if strike_type in ["Duck", "Slip Left", "Slip Right"]:
            note["type"] = "defense"
        self.choreography.append(note)
        self.choreography.sort(key=lambda x: x['timestamp'])
        self.redraw_choreography()
        print(f"Added: {strike_type} at {timestamp:.2f}s")

    def delete_choreography_note(self, timestamp):
        # ... (same as before)
        closest_note = None
        min_dist = float('inf')
        for note in self.choreography:
            dist = abs(note['timestamp'] - timestamp)
            if dist < 0.1 and dist < min_dist:
                min_dist = dist
                closest_note = note
        if closest_note:
            self.choreography.remove(closest_note)
            self.redraw_choreography()
            print(f"Deleted note at {closest_note['timestamp']:.2f}s")

    def redraw_choreography(self):
        # ... (same as before)
        for marker in self.choreography_markers:
            marker.remove()
        self.choreography_markers.clear()
        for note in self.choreography:
            color = 'm' if note['type'] == 'strike' else 'c'
            marker = self.ax.axvline(note['timestamp'], color=color, lw=2, linestyle='-')
            self.choreography_markers.append(marker)
        self.canvas.draw()

    def export_level(self):
        # ... (same as before)
        if not self.audio_file_path:
            tk.messagebox.showwarning("Warning", "No audio file loaded.")
            return
        output_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Level As...")
        if not output_path:
            return
        song_duration = librosa.get_duration(y=self.audio_data, sr=self.sampling_rate)
        output_data = {
            "song_filename": self.audio_file_path.split('/')[-1],
            "song_duration_seconds": song_duration,
            "estimated_bpm": self.analysis_results.get('estimated_bpm', 0),
            "analysis_data": {
                k: self.analysis_results.get(k, []) for k in ['beats', 'onsets', 'percussive_onsets']
            },
            "choreography": self.choreography
        }
        try:
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            tk.messagebox.showinfo("Success", f"Level exported successfully to {output_path}")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to export level: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    # It's good practice to initialize pygame early.
    pygame.init()
    app = RhythmKickboxingLevelEditor(root)
    root.mainloop()
    pygame.quit()
