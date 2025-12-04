import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import time
import matplotlib.pyplot as plt
from collections import Counter
import vlc
import random
import os
import tkinter as tk
from tkinter import Label, Button
import threading
import ctypes

# Load pre-trained emotion detection model
print("Loading emotion detection model...")
model = tf.keras.models.load_model('emotion_model.h5')
print("Model loaded successfully.")

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Define song playlists for each emotion
emotion_songs = {
    'Happy': ['ye_ga_ye_maina.mp3'],
    'Sad': ['TADAP_TADAP.mp3'],
    'Surprised': ['Aake teri baahon mein.mp3'],
    'Fear': ['Shirdi_wale_SaiBaba.mp3'],
    'Disgusted': ['galavar_khali.mp3'], 
    'Neutral': ['Chand chupa badal mein.mp3'],
    'Angry': ['tik_tik_vajate_sad.mp3']
}

# Initialize MTCNN detector
detector = MTCNN()

# Ensure VLC works on Windows
if os.name == "nt":
    vlc_path = r"C:\Program Files\VideoLAN\VLC"  # Update this if VLC is installed in another location
    os.add_dll_directory("C:\Program Files\VideoLAN\VLC")
    ctypes.CDLL(os.path.join("C:\Program Files\VideoLAN\VLC", "libvlc.dll"))

# Emotion detection variables
cap = None
detection_running = False
emotion_history = []
player = None  # VLC player instance

def detect_emotion(frame):
    faces = detector.detect_faces(frame)
    if faces:
        x, y, w, h = faces[0]['box']
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)
        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]
        return emotion, (x, y, w, h)
    return None, None

def play_song(emotion):
    global player
    if emotion in emotion_songs:
        song = random.choice(emotion_songs[emotion])
        print(f"Playing song: {song}")
        player = vlc.MediaPlayer(song)
        player.play()

def pause_song():
    if player:
        player.pause()

def resume_song():
    if player:
        player.play()

def stop_song():
    if player:
        player.stop()

def start_detection():
    global cap, detection_running, emotion_history
    detection_running = True
    cap = cv2.VideoCapture(0)
    emotion_history.clear()
    
    def run_detection():
        while detection_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            emotion, bbox = detect_emotion(frame)
            if emotion:
                emotion_history.append(emotion)
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if emotion_history:
            most_common_emotion = Counter(emotion_history).most_common(1)[0][0]
            play_song(most_common_emotion)
            show_graphs()
    
    threading.Thread(target=run_detection, daemon=True).start()

def stop_detection():
    global detection_running
    detection_running = False
    print("Detection stopped.")

def show_graphs():
    emotion_counts = Counter(emotion_history)
    labels, values = zip(*emotion_counts.items())
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    plt.title('Emotion Distribution')
    
    plt.subplot(1, 2, 2)
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Emotions')
    plt.ylabel('Count')
    plt.title('Emotion Frequency')
    
    plt.show()

# GUI setup
root = tk.Tk()
root.title("Emotion Detection & Music Player")
root.geometry("600x400")

quote_label = Label(root, text="Stay positive, work hard, make it happen!", font=("Arial", 14))
quote_label.pack(pady=10)

start_button = Button(root, text="Start Detection", command=start_detection)
start_button.pack(pady=5)

stop_button = Button(root, text="Stop Detection", command=stop_detection)
stop_button.pack(pady=5)

pause_button = Button(root, text="Pause Song", command=pause_song)
pause_button.pack(pady=5)

resume_button = Button(root, text="Resume Song", command=resume_song)
resume_button.pack(pady=5)

stop_song_button = Button(root, text="Stop Song", command=stop_song)
stop_song_button.pack(pady=5)

root.mainloop()
