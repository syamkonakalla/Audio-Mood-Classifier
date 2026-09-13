import streamlit as st
import librosa
import joblib
import pandas as pd
import whisper
transcript_model = whisper.load_model("base")
model = joblib.load('trained_mood_model.pkl')
scaler = joblib.load('scaler.pkl')
songs_data=pd.read_csv('labeled_songs_with_moods.csv')


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
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    
   
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    loudness = librosa.feature.rms(y=y).mean()  # Use .mean() to flatten the array to a scalar
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key = chroma.argmax(axis=0).mean()  # Flatten chroma and extract mean value
    mode = 1 if chroma.mean() > 0.5 else 0
    duration_ms = librosa.get_duration(y=y, sr=sr) * 1000
    danceability = (tempo / 200) + (1 - librosa.feature.zero_crossing_rate(y).mean())
    energy = (y ** 2).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()  # Scalar value
    speechiness = spectral_rolloff / sr
    acousticness = librosa.feature.spectral_contrast(y=y, sr=sr).mean()  # Scalar value
    harmonic, _ = librosa.effects.hpss(y)
    instrumentalness = harmonic.mean()  # Scalar value
    liveness = librosa.feature.spectral_flatness(y=y).mean()  # Scalar value
    valence = 0.5  # Placeholder for valence (requires model)
    time_signature = 4  # Default assumption

    # Create a list of features in the specified order
    features_list = [
        danceability[0],
        energy,
        int(key),
        loudness,
        mode,
        speechiness,
        acousticness,
        instrumentalness,
        liveness,
        valence,
        tempo[0],
        duration_ms,
        time_signature,
    ]
    
    return features_list

# Streamlit UI
st.title("Audio Mood Classifier App")

# Upload audio file
input_method = st.radio(
    "Choose your input method:",
    ("Upload Audio File", "Select Song by name")
)
if input_method == "Upload Audio File":
        
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])

    if uploaded_file is not None:
        # Save uploaded file temporarily
        if st.button("Extract Features and Predict Mood"):
            with open("temp_audio_file", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract features
            features = extract_audio_features("temp_audio_file")
            result=predict_mood(features)
            st.subheader("Mood Prediction")
            st.write(f"The predicted mood of the song is: **{result}**")  # Print the result in Streamlit
            st.subheader("suggested songs")
            sugested_songs=songs_data[songs_data['Mood'] == result]
            st.write(sugested_songs['song_title'])
        if st.button("Lyric of the Song"):
            result = transcript_model.transcribe("temp_audio_file")
            st.subheader("Lyric")
            st.write(result["text"])

elif input_method == "Select Song by name":
    song_name = st.text_input("Search for a song", "")
    if st.button("Continue"):
        
        features=pd.read_csv("data.csv")
        feature_values=features.loc[features['song_title'] == song_name, ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']]
        row = feature_values.iloc[0]
        feature_values = row.tolist()
        result=predict_mood(feature_values)
        st.subheader("Mood Prediction")
        st.write(f"The predicted mood of the song is: **{result}**")  # Print the result in Streamlit
        st.subheader("suggested songs")
        sugested_songs=songs_data[songs_data['Mood'] == result]
        st.write(sugested_songs['song_title'])
        
        

