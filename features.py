import librosa

# Load the audio file
audio_file_path = 'song.mp3'
y, sr = librosa.load(audio_file_path, sr=None)

# Feature Extraction
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
loudness = librosa.feature.rms(y=y).mean()
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
key = chroma.argmax(axis=0).mean()
mode = 1 if chroma.mean() > 0.5 else 0
duration_ms = librosa.get_duration(y=y, sr=sr) * 1000
danceability = (tempo / 200) + (1 - librosa.feature.zero_crossing_rate(y).mean())
energy = (y ** 2).mean()
spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
speechiness = spectral_rolloff / sr
acousticness = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
harmonic, _ = librosa.effects.hpss(y)
instrumentalness = harmonic.mean()
liveness = librosa.feature.spectral_flatness(y=y).mean()
valence = 0.5  # Placeholder for valence (needs ML model for accurate prediction)
time_signature = 4

# Reorganize features in the specified order
features = {
    "danceability": danceability,
    "energy": energy,
    "key": int(key),
    "loudness": loudness,
    "mode": mode,
    "speechiness": speechiness,
    "acousticness": acousticness,
    "instrumentalness": instrumentalness,
    "liveness": liveness,
    "valence": valence,
    "tempo": tempo,
    "duration_ms": duration_ms,
    "time_signature": 4,  # Assumed as default for this example
}
features_list = [
    danceability,
    energy,
    int(key),
    loudness,
    mode,
    speechiness,
    acousticness,
    instrumentalness,
    liveness,
    valence,
    tempo,
    duration_ms,
    time_signature,
]
# Display in the specified order
for key, value in features.items():
    print(key,value)
