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
import ctypes

# Load pre-trained emotion detection model
print("Loading emotion detection model...")
model = tf.keras.models.load_model('emotion_model.h5')
print("Model loaded successfully.")

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Define song playlists for each emotion
emotion_songs = {
    'Happy': ['Chand chupa badal mein.mp3'],
    'Sad': ['TADAP_TADAP.mp3'],
    'Surprised': ['Aake teri baahon mein.mp3'],
    'Fear': ['Shirdi_wale_SaiBaba.mp3'],
    'Disgusted': ['ye_ga_ye_maina.mp3'],
    'Neutral': ['galavar_khali.mp3'],
    'Angry': ['tik_tik_vajate_sad.mp3']
}

# Initialize MTCNN detector
print("Initializing MTCNN detector...")
detector = MTCNN()
print("MTCNN initialized.")

# Ensure VLC works on Windows
if os.name == "nt":
    vlc_path = r"C:\Program Files\VideoLAN\VLC"  # Update this if VLC is installed in another location
    os.add_dll_directory("C:\Program Files\VideoLAN\VLC")
    ctypes.CDLL(os.path.join("C:\Program Files\VideoLAN\VLC", "libvlc.dll"))

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
    if emotion in emotion_songs:
        song = random.choice(emotion_songs[emotion])
        print(f"Playing song: {song}")
        player = vlc.MediaPlayer(song)
        player.play()
        while player.get_state() != vlc.State.Ended:
            time.sleep(1)

def main():
    cap = cv2.VideoCapture(0)
    emotion_history = []
    detection_start_time = time.time()
    detection_duration = 10  # Detect emotions for 10 seconds
    
    while time.time() - detection_start_time < detection_duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        emotion, bbox = detect_emotion(frame)
        if emotion:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            emotion_history.append(emotion)
        
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if emotion_history:
        most_common_emotion = Counter(emotion_history).most_common(1)[0][0]
        print(f"Final detected emotion: {most_common_emotion}")
        play_song(most_common_emotion)
    else:
        print("No emotions detected.")

if __name__ == '__main__':
    main()
