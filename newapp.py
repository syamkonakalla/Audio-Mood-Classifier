import streamlit as st
import librosa
import joblib
import pandas as pd
import whisper

# Load models and data
transcript_model = whisper.load_model("base")
model = joblib.load('trained_mood_model.pkl')
scaler = joblib.load('scaler.pkl')
songs_data = pd.read_csv('labeled_songs_with_moods.csv')

# Function to predict mood
def predict_mood(features):
    if features:
        # Scale the features
        features_scaled = scaler.transform([features])
        # Predict mood
        predicted_mood = model.predict(features_scaled)[0]
        return predicted_mood
    else:
        return "Could not retrieve features for prediction."

# Function to extract features
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    
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
    valence = 0.5
    time_signature = 4

    return [
        danceability, energy, int(key), loudness, mode,
        speechiness, acousticness, instrumentalness,
        liveness, valence, tempo, duration_ms, time_signature
    ]

# Streamlit UI setup
st.set_page_config(page_title="Audio Mood Classifier", layout="centered", page_icon="🎵")

st.title("🎧 Audio Mood Classifier")
st.markdown(
    """
    **Discover the mood of your favorite songs!**  
    Upload an audio file or select a song by name to analyze its features, predict its mood, and get song recommendations.
    """
)

# Input method selection
input_method = st.radio(
    "Choose your input method:",
    ["Upload Audio File", "Search Song by Name"],
    index=0
)

if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader("🎵 Upload your audio file (mp3, wav, ogg):", type=["mp3", "wav", "ogg"])
    
    if uploaded_file and st.button("Extract Features and Predict Mood"):
        with open("temp_audio_file", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract features
        features = extract_audio_features("temp_audio_file")
        predicted_mood = predict_mood(features)

        # Display mood
        st.subheader("🎭 Mood Prediction")
        st.success(f"The predicted mood is: **{predicted_mood}**")
        
        # Display suggested songs
        st.subheader("🎶 Suggested Songs")
        suggested_songs = songs_data[songs_data['Mood'] == predicted_mood]
        st.write(suggested_songs[['song_title', 'artist']])
        
        # Transcribe lyrics
        if st.button("Show Lyrics"):
            result = transcript_model.transcribe("temp_audio_file")
            st.subheader("📝 Lyrics")
            st.write(result["text"])

elif input_method == "Search Song by Name":
    song_name = st.text_input("🔍 Enter the name of a song:")
    
    if st.button("Find Mood"):
        # Retrieve features from dataset
        features_data = pd.read_csv("data.csv")
        feature_row = features_data.loc[features_data['song_title'] == song_name, [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'duration_ms', 'time_signature'
        ]]
        
        if not feature_row.empty:
            features = feature_row.iloc[0].tolist()
            predicted_mood = predict_mood(features)
            
            # Display mood
            st.subheader("🎭 Mood Prediction")
            st.success(f"The predicted mood is: **{predicted_mood}**")
            
            # Display suggested songs
            st.subheader("🎶 Suggested Songs")
            suggested_songs = songs_data[songs_data['Mood'] == predicted_mood]
            st.write(suggested_songs[['song_title', 'artist']])
        else:
            st.error("⚠️ Song not found in the database.")
