import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pygame
import soundfile as sf
import tempfile
import os
import atexit
from audio_analyzer import analyze_audio

class RhythmKickboxingLevelEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Rhythm Kickboxing Level Editor")
        self.root.geometry("1200x800")

        # Core data
        self.audio_file_path = None
        self.audio_data = None
        self.sampling_rate = None
        self.analysis_results = {}
        self.choreography = []

        # Playback state
        self.is_playing = False
        self.is_paused = False
        self.current_playback_rate = 1.0
        self.temp_audio_file = None

        # UI state
        self.is_separated_view = False

        # UI elements
        self.playback_position_marker = None
        self.choreography_markers = []
        self.ax = None
        self.axes = []

        self.strike_types = [
            "Left Jab", "Right Cross", "Lead Hook", "Rear Hook",
            "Lead Uppercut", "Rear Uppercut", "Duck", "Slip Left", "Slip Right"
        ]

        atexit.register(self.cleanup)

        # UI Setup
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        controls_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.setup_controls(controls_frame)

        self.visualization_frame = ttk.LabelFrame(main_frame, text="Song Map", padding="10")
        self.visualization_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)

        self.fig = plt.figure(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.visualization_frame)

        toolbar = NavigationToolbar2Tk(self.canvas, self.visualization_frame)
        toolbar.update()

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

    def setup_controls(self, parent_frame):
        file_ops_frame = ttk.Frame(parent_frame)
        file_ops_frame.pack(fill=tk.X, pady=5)

        self.upload_btn = ttk.Button(file_ops_frame, text="Upload Song", command=self.upload_song)
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        self.analyze_btn = ttk.Button(file_ops_frame, text="Analyze Music", command=self.analyze_music, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        self.export_btn = ttk.Button(file_ops_frame, text="Export Level", command=self.export_level, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=5)
        self.separate_btn = ttk.Button(file_ops_frame, text="Separate Timelines", command=self.toggle_timeline_view, state=tk.DISABLED)
        self.separate_btn.pack(side=tk.LEFT, padx=5)

        analysis_options_frame = ttk.LabelFrame(parent_frame, text="Analysis Options")
        analysis_options_frame.pack(fill=tk.X, pady=10, ipady=5)
        self.analysis_vars = {
            "BPM & Beats": tk.BooleanVar(value=True),
            "Onsets (Transients)": tk.BooleanVar(value=True),
            "Energy (RMS)": tk.BooleanVar(value=True),
            "Percussive Elements": tk.BooleanVar(value=True)
        }
        for text, var in self.analysis_vars.items():
            ttk.Checkbutton(analysis_options_frame, text=text, variable=var).pack(side=tk.LEFT, padx=10)

        playback_frame = ttk.Frame(parent_frame)
        playback_frame.pack(fill=tk.X, pady=5)
        self.play_pause_btn = ttk.Button(playback_frame, text="Play", command=self.toggle_playback, state=tk.DISABLED)
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(playback_frame, text="Stop", command=self.stop_audio, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        speed_frame = ttk.Frame(parent_frame)
        speed_frame.pack(fill=tk.X, pady=5)
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT, padx=5)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_scale = ttk.Scale(speed_frame, from_=0.5, to=2.0, orient=tk.HORIZONTAL, variable=self.speed_var, command=self.update_speed_label)
        self.speed_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.speed_label = ttk.Label(speed_frame, text="1.0x")
        self.speed_label.pack(side=tk.LEFT, padx=5)

    def update_speed_label(self, value):
        self.speed_label.config(text=f"{float(value):.2f}x")

    def cleanup(self):
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):
            os.remove(self.temp_audio_file)
            self.temp_audio_file = None

    def upload_song(self):
        self.cleanup()
        self.audio_file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.ogg *.flac")])
        if not self.audio_file_path: return

        try:
            if pygame.mixer.get_init(): pygame.mixer.music.stop()
            self.audio_data, self.sampling_rate = librosa.load(self.audio_file_path, sr=None)
            pygame.mixer.quit()
            pygame.mixer.init(frequency=self.sampling_rate)

            self.plot_waveform()
            for btn in [self.analyze_btn, self.play_pause_btn, self.stop_btn, self.export_btn, self.separate_btn]:
                btn['state'] = tk.NORMAL
            self.is_playing = False
            self.is_paused = False
            self.play_pause_btn.config(text="Play")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio file: {e}")

    def plot_waveform(self):
        self.fig.clear()
        if self.is_separated_view:
            gs_kw = dict(height_ratios=[3] + [1] * len(self.strike_types))
            self.axes = self.fig.subplots(nrows=len(self.strike_types) + 1, ncols=1, sharex=True, gridspec_kw=gs_kw)
            self.ax = self.axes[0]
            librosa.display.waveshow(self.audio_data, sr=self.sampling_rate, ax=self.ax, alpha=0.5)
            self.ax.set_title("Song Waveform")

            for i, strike_type in enumerate(self.strike_types):
                self.axes[i+1].set_ylabel(strike_type, rotation=0, ha='right', va='center', fontsize=8)
                self.axes[i+1].set_yticks([])
                self.axes[i+1].grid(True, which='major', axis='x', linestyle='--')
            plt.xlabel("Time (s)")
        else:
            self.ax = self.fig.add_subplot(111)
            self.axes = [self.ax]
            librosa.display.waveshow(self.audio_data, sr=self.sampling_rate, ax=self.ax, alpha=0.5)
            self.ax.set_title("Song Waveform")
            self.ax.set_xlabel("Time (s)")

        self.ax.set_ylabel("Amplitude")
        self.canvas.draw()

    def analyze_music(self):
        if not hasattr(self, 'ax') or self.ax is None: return
        print("Analyzing music...")
        analysis_options = {key: var.get() for key, var in self.analysis_vars.items()}
        self.analysis_results = analyze_audio(self.audio_data, self.sampling_rate, analysis_options)

        # Clear previous analysis visuals
        for line in self.ax.lines: line.remove()
        for vline in self.ax.collections: vline.remove()

        if "beats" in self.analysis_results:
            self.ax.vlines(self.analysis_results['beats'], -1, 1, color='b', linestyle='--', label=f"Beats (BPM: {self.analysis_results.get('estimated_bpm', 0):.2f})")
        if "onsets" in self.analysis_results:
            self.ax.vlines(self.analysis_results['onsets'], -1, 1, color='r', linestyle=':', label='Onsets')
        if "rms_energy" in self.analysis_results:
            rms_data = self.analysis_results['rms_energy']
            times = [p[0] for p in rms_data]
            rms_values = np.array([p[1] for p in rms_data])
            if rms_values.size > 0:
                rms_normalized = rms_values / np.max(rms_values) if np.max(rms_values) > 0 else rms_values
                self.ax.plot(times, rms_normalized, color='g', alpha=0.6, label='Energy (RMS)')
        if "percussive_onsets" in self.analysis_results:
            self.ax.vlines(self.analysis_results['percussive_onsets'], -1, 1, color='y', linestyle='-.', label='Percussive Onsets')

        self.ax.legend()
        self.canvas.draw()
        print("Analysis complete.")

    def toggle_playback(self):
        rate = self.speed_var.get()
        if not self.is_playing:
            self.current_playback_rate = rate
            file_to_play = self.audio_file_path
            if self.current_playback_rate != 1.0:
                try:
                    stretched_audio = librosa.effects.time_stretch(self.audio_data, rate=self.current_playback_rate)
                    self.cleanup()
                    fd, self.temp_audio_file = tempfile.mkstemp(suffix=".wav")
                    os.close(fd)
                    sf.write(self.temp_audio_file, stretched_audio, self.sampling_rate)
                    file_to_play = self.temp_audio_file
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to create time-stretched audio: {e}")
                    return

            pygame.mixer.music.load(file_to_play)
            pygame.mixer.music.play()
            self.is_playing = True
            self.is_paused = False
            self.play_pause_btn.config(text="Pause")
            self.update_playback_marker()
        elif self.is_playing and not self.is_paused:
            pygame.mixer.music.pause()
            self.is_paused = True
            self.play_pause_btn.config(text="Resume")
        elif self.is_playing and self.is_paused:
            pygame.mixer.music.unpause()
            self.is_paused = False
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
            current_time_stretched = pygame.mixer.music.get_pos() / 1000.0
            current_time_original = current_time_stretched / self.current_playback_rate

            if self.playback_position_marker: self.playback_position_marker.remove()
            self.playback_position_marker = self.ax.axvline(current_time_original, color='purple', lw=2)
            self.canvas.draw_idle()
            self.root.after(50, self.update_playback_marker)
        elif not pygame.mixer.music.get_busy() and self.is_playing:
            self.stop_audio()

    def on_canvas_click(self, event):
        clicked_ax = None
        for ax in self.axes:
            if event.inaxes == ax:
                clicked_ax = ax
                break
        if clicked_ax is None: return
        timestamp = event.xdata
        if event.button == 1: self.show_choreography_menu(event.x, event.y, timestamp)
        elif event.button == 3: self.delete_choreography_note(timestamp)

    def show_choreography_menu(self, x, y, timestamp):
        menu = tk.Menu(self.root, tearoff=0)
        for strike in self.strike_types:
            menu.add_command(label=strike, command=lambda s=strike: self.add_choreography_note(timestamp, s))
        try:
            menu.tk_popup(x, self.root.winfo_y() + y, 0)
        finally:
            menu.grab_release()

    def add_choreography_note(self, timestamp, strike_type):
        note = {"timestamp": timestamp, "type": "strike", "value": strike_type}
        if strike_type in ["Duck", "Slip Left", "Slip Right"]: note["type"] = "defense"
        self.choreography.append(note)
        self.choreography.sort(key=lambda x: x['timestamp'])
        self.redraw_choreography()
        print(f"Added: {strike_type} at {timestamp:.2f}s")

    def delete_choreography_note(self, timestamp):
        closest_note, min_dist = None, float('inf')
        for note in self.choreography:
            dist = abs(note['timestamp'] - timestamp)
            if dist < 0.1 and dist < min_dist:
                min_dist, closest_note = dist, note
        if closest_note:
            self.choreography.remove(closest_note)
            self.redraw_choreography()
            print(f"Deleted note at {closest_note['timestamp']:.2f}s")

    def toggle_timeline_view(self):
        self.is_separated_view = not self.is_separated_view
        self.separate_btn.config(text="Combine Timelines" if self.is_separated_view else "Separate Timelines")
        self.plot_waveform()
        if self.analysis_results: self.analyze_music()
        self.redraw_choreography()

    def redraw_choreography(self):
        for marker in self.choreography_markers: marker.remove()
        self.choreography_markers.clear()

        if self.is_separated_view:
            for note in self.choreography:
                try:
                    strike_index = self.strike_types.index(note['value'])
                    ax_to_plot_on = self.axes[strike_index + 1]
                    color = 'm' if note['type'] == 'strike' else 'c'
                    marker = ax_to_plot_on.axvline(note['timestamp'], color=color, lw=2)
                    self.choreography_markers.append(marker)
                except (ValueError, IndexError):
                    marker = self.axes[0].axvline(note['timestamp'], color='gray', lw=1, linestyle=':')
                    self.choreography_markers.append(marker)
        else:
            for note in self.choreography:
                color = 'm' if note['type'] == 'strike' else 'c'
                marker = self.ax.axvline(note['timestamp'], color=color, lw=2, linestyle='-')
                self.choreography_markers.append(marker)
        self.canvas.draw()

    def export_level(self):
        if not self.audio_file_path:
            messagebox.showwarning("Warning", "No audio file loaded.")
            return
        output_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], title="Save Level As...")
        if not output_path: return

        song_duration = librosa.get_duration(y=self.audio_data, sr=self.sampling_rate)
        output_data = {
            "song_filename": os.path.basename(self.audio_file_path),
            "song_duration_seconds": song_duration,
            "estimated_bpm": self.analysis_results.get('estimated_bpm', 0),
            "analysis_data": {k: self.analysis_results.get(k, []) for k in ['beats', 'onsets', 'percussive_onsets']},
            "choreography": self.choreography
        }
        try:
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            messagebox.showinfo("Success", f"Level exported successfully to {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export level: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    pygame.init()
    app = RhythmKickboxingLevelEditor(root)
    root.mainloop()
    pygame.quit()
