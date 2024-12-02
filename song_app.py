import base64
import librosa
import joblib
import pandas as pd
import whisper
import os
import tempfile
import streamlit as st
import time
import warnings

# Load models and data
@st.cache_resource
def load_models():
    transcript_model = whisper.load_model("base")
    model = joblib.load('trained_mood_model.pkl')
    scaler = joblib.load('scaler.pkl')
    songs_data = pd.read_csv('labeled_songs.csv')
    return transcript_model, model, scaler, songs_data

transcript_model, model, scaler, songs_data = load_models()

# Function to predict mood
def predict_mood(features):
    if features:
        features_scaled = scaler.transform([features])
        predicted_mood = model.predict(features_scaled)[0]
        return predicted_mood
    else:
        return "Could not retrieve features for prediction."

# Function to extract features
def extract_audio_features(audio_path):
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

# Function to set the background image
def set_page_background(png_file):
    @st.cache_data()
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
        }}
        body {{
            font-family: "Arial", sans-serif;
            font-size: 18px;
            font-weight: bold;
        }}
        .stMarkdown {{
            font-size: 24px;
            font-weight: bold;
        }}
        .stButton {{
            font-size: 20px;
            font-weight: bold;
        }}
        .stTextInput {{
            font-size: 20px;
            font-weight: bold;
        }}
        .stSubheader {{
            font-size: 30px;
            font-weight: bold;
        }}
        .stTitle {{
            font-size: 36px;
            font-weight: bold;
        }}
        </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Path to your images
background_image_path = r"pexels-pixabay-534283.jpg"
sidebar_image_path = r"silhouette-musical-note-clef-b.jpg"

# Call the set_page_background function with the image path
set_page_background(background_image_path)

# Sidebar for navigation (with dropdown)
st.sidebar.image(sidebar_image_path, use_container_width=True)
st.sidebar.title("Explore!")

sidebar_option = st.sidebar.selectbox(
    "Select a feature:",
    ["HomePage", "Audio Mood Prediction", "MoodMatch", "Lyricify", "HappyVibes"]
)
st.sidebar.markdown("---")  # Separator
st.sidebar.subheader("🎉 About This App")
st.sidebar.markdown(
    """
    **Smart Music Mood Classification System**  
    Built with ❤️ by creative minds who believe music has the power to transform moods.
    
    Have a question or suggestion?  
    📧 Email us at:  
    [AshwinMuralidharan@my.unt.edu](mailto:AshwinMuralidharan@my.unt.edu)
    
    🎵 "Music is life, that's why our hearts have beats!" 🎶
    """
)
# Main page layout
if sidebar_option == "HomePage":
    st.title("Welcome to the Smart Music Mood Classification System")
    st.image("Banner.png", use_container_width=True)
    st.subheader("Project Overview")
    st.markdown(
        """
        **Want to create a playlist for every mood?**  
        Use our Smart Music Mood Classification System to find out what mood all your favorite songs fit into.

        ### Modules:
        1. **Audio Mood Prediction**:
            - Upload an audio file to predict its mood.
            - Features include audio feature extraction, mood classification, and lyric transcription.

        2. **MoodMatch**:
            - Search for a song by its name.
            - Predicts the song's mood and provides song recommendations in the same mood.

        3. **Lyricify**:
            - Generate lyrics for a song based on its name.
            - Enables creativity and songwriting assistance.

        4. **HappyVibes**:
            - Provides mood-based song recommendations.
            - Select a mood (e.g., Happy, Sad, Energetic, Relaxing) and get a curated list of songs matching the mood.

        ### Developers:
        - **Ashwin Muralidharan**  
          Email: [AshwinMuralidharan@my.unt.edu](mailto:AshwinMuralidharan@my.unt.edu)
        - **Manideep Sharma Domudala**  
          Email: [ManideepSharmaDomudala@my.unt.edu](mailto:ManideepSharmaDomudala@my.unt.edu)
        - **Syam Sai Konakalla**  
          Email: [SyamSaiKonakalla@my.unt.edu](mailto:SyamSaiKonakalla@my.unt.edu)
        - **Alessandra Palladino**  
          Email: [alessandrapalladinoromero@my.unt.edu](mailto:alessandrapalladinoromero@my.unt.edu)
        """
    )

elif sidebar_option == "Audio Mood Prediction":
    st.title("Audio Mood Prediction")
    st.subheader("Upload and Analyze Your Sound")
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

elif sidebar_option == "MoodMatch":
    st.title("MoodMatch")
    st.subheader("Search Songs by Name for Mood Prediction")
    song_name = st.text_input("🔍 Enter the name of a song:")
    
    if st.button("Find Mood"):
        feature_row = songs_data.loc[songs_data['song_title'] == song_name, [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'duration_ms', 'time_signature'
        ]] 

        if not feature_row.empty:
            features = feature_row.iloc[0].tolist()
            predicted_mood = predict_mood(features)
            
            st.subheader("🎭 Mood Prediction")
            st.success(f"The predicted mood is: **{predicted_mood}**")
            
            st.subheader("🎶 Suggested Songs")
            suggested_songs = songs_data[songs_data['Mood'] == predicted_mood]
            st.write(suggested_songs[['song_title', 'artist']])
        else:
            st.error("⚠️ Song not found in the database.")

elif sidebar_option == "Lyricify":
    st.title("Lyricify")
    st.subheader("Generate Lyrics for a Song")
    uploaded_file = st.file_uploader("🎵 Upload your audio file (mp3, wav, ogg):", type=["mp3", "wav", "ogg"])
    if st.button("Show Lyrics"):
        with open("temp_audio_file", "wb") as f:
            f.write(uploaded_file.getbuffer())
        result = transcript_model.transcribe("temp_audio_file")
        st.subheader("📝 Lyrics")
        st.write(result["text"])

elif sidebar_option == "HappyVibes":
    st.title("HappyVibes")
    st.subheader("Mood-based Song Recommendations")
    mood = st.selectbox("🎵 Select a mood:", ["Happy", "Sad", "Energetic", "Relaxing"])
    
    if st.button("Find Songs"):
        try:
            recommended_songs = songs_data[songs_data['Mood'] == mood]
            st.subheader(f"🎶 Songs for {mood} Mood")
            st.write(recommended_songs[['song_title', 'artist']])
        except Exception as e:
            st.error(f"Error fetching songs: {str(e)}")

