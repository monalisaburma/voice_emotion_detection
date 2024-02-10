import tkinter as tk
from tkinter import filedialog
import sounddevice as sd
import numpy as np
import librosa
from keras.models import model_from_json

class EmotionDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Emotion Detection App")

        self.title_label = tk.Label(self.master, text="Voice Emotion Recognition", font=('Helvetica', 16, 'bold'))
        self.title_label.pack(pady=10)

        self.upload_button = tk.Button(self.master, text="Upload Voice File", command=self.upload_file, font=('Helvetica', 14))
        self.upload_button.pack(pady=20)

        self.record_button = tk.Button(self.master, text="Start Recording", command=self.start_recording, font=('Helvetica', 14))
        self.record_button.pack(pady=20)

        self.recording = False

    def upload_file(self):
        file_path = filedialog.askopenfilename(
            title="Select an Audio File",
            filetypes=[
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("All files", "*.*")  
    ]
)
        if file_path:
            emotion = self.detect_emotion(file_path)
            self.display_result(emotion)

    def start_recording(self):
        self.recording = not self.recording
        if self.recording:
            self.record_button.config(text="Stop Recording")
            self.record_audio()
        else:
            self.record_button.config(text="Start Recording")

    def record_audio(self):
        duration = 3  
        sampling_rate = 44100
        audio_data = sd.rec(int(sampling_rate * duration), samplerate=sampling_rate, channels=1, dtype='int16')
        sd.wait()

        features = self.extract_features_realtime(audio_data, sampling_rate)

        emotion = self.detect_emotion_realtime(features)
        self.display_result(emotion)

    def detect_emotion_realtime(self, features):
        with open('best_model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('best_model1.h5')

        prediction = loaded_model.predict(np.expand_dims(features, axis=0))
        emotion_label = self.get_emotion_label(np.argmax(prediction))

        return emotion_label
    
    def detect_emotion(self, file_path):
        try:
            features = self.extract_features(file_path)  
            emotion = self.detect_emotion_realtime(features)  
            return emotion
        except Exception as e:
            print(f"Error during emotion detection: {str(e)}")
            return "Unknown"
        
    def extract_features(self, file_path):
        y, sr = librosa.load(file_path, duration=3)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.expand_dims(np.mean(mfccs, axis=1), axis=-1)


    def extract_features_realtime(self, audio_data, sampling_rate):
        y = audio_data.flatten().astype(np.float32)
        mfccs = librosa.feature.mfcc(y=y, sr=sampling_rate, n_mfcc=40)
        return np.expand_dims(np.mean(mfccs, axis=1), axis=-1)

    def get_emotion_label(self, emotion_index):
        emotion_labels = ['fear', 'angry', 'disgust', 'neutral', 'sad', 'ps', 'happy']
        return emotion_labels[emotion_index]

    def display_result(self, emotion):
        result_label = tk.Label(self.master, text=f"Detected Emotion: {emotion}", font=('Helvetica', 14))
        result_label.pack(pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.geometry("600x400")  
    root.mainloop()
