import streamlit as st
import librosa
import joblib
import pandas as pd
import whisper
import base64

st.set_page_config(page_title="Audio Mood Classifier", layout="centered", page_icon="🎵")

# Add background image styling
def set_background_image(image_file):
    with open(image_file, "rb") as f:
        image_data = f.read()
    encoded_image = base64.b64encode(image_data).decode()
    background_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_image}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

set_background_image("pexels-pixabay-534283.jpg")  # Replace with the path to your background image

# Add consistent styling
st.markdown("""
    <style>
    .stTitle, .stHeader {{
        font-size: 36px;
        font-weight: bold;
        color: #333333;
    }}
    .stSidebar {{
        background-color: #f5f5f5;
    }}
    .stMarkdown, .stTextInput {{
        font-size: 18px;
        color: #444444;
    }}
    .stButton {{
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
    }}
    </style>
""", unsafe_allow_html=True)
# Sidebar for navigation
st.sidebar.image("silhouette-musical-note-clef-b.jpg")
st.sidebar.title("Explore!")

sidebar_option = st.sidebar.selectbox(
    "Select a feature:",
    ["HomePage","Audio Mood Prediction", "MoodMatch", "🎧 Lyricify", "HappyVibes"]
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
    [SyamSaiKonakalla@my.unt.edu](mailto:SyamSaiKonakalla@my.unt.edu)
    [ManideepSharmaDomudala@my.unt.edu](mailto:ManideepSharmaDomudala@my.unt.edu)
    [alessandrapalladinoromero@my.unt.edu](mailto:alessandrapalladinoromero@my.unt.edu)
    
    🎵 "Music is life, that's why our hearts have beats!" 🎶
    """
)
# Load models and data

transcript_model = whisper.load_model("base")
model = joblib.load('trained_mood_model.pkl')
scaler = joblib.load('scaler.pkl')
songs_data = pd.read_csv('labeled_songs.csv')

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

# Streamlit UI setup



# Main page layout
st.title("🎧 Audio Mood Classifier")
st.markdown(
    """
    **Discover the mood of your favorite songs!**  
    Upload an audio file or select a song by name to analyze its features, predict its mood, and get song recommendations.
    """
)
if sidebar_option == "HomePage":
    st.title("Welcome to the Smart Music Mood Classification System")
    st.image("Banner.png")
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
        - **Syam Sai Konakalla**  
          Email: [SyamSaiKonakalla@my.unt.edu](mailto:SyamSaiKonakalla@my.unt.edu)
        - **Manideep Sharma Domudala**  
          Email: [ManideepSharmaDomudala@my.unt.edu](mailto:ManideepSharmaDomudala@my.unt.edu)
        - **Alessandra Palladino**  
          Email: [alessandrapalladinoromero@my.unt.edu](mailto:alessandrapalladinoromero@my.unt.edu)
        """
    )
elif sidebar_option == "Audio Mood Prediction":
    # Upload audio file
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
            with open("temp_audio_file", "wb") as f:
                f.write(uploaded_file.getbuffer())
            result = transcript_model.transcribe("temp_audio_file")
            st.subheader("📝 Lyrics")
            st.write(result["text"])

elif sidebar_option == "MoodMatch":
    song_name = st.text_input("🔍 Enter the name of a song:")
    
    if st.button("Find Mood"):
        # Retrieve features from dataset
        features_data = pd.read_csv("labeled_songs.csv")
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
elif sidebar_option == "🎧 Lyricify":
    uploaded_file = st.file_uploader("🎵 Upload your audio file (mp3, wav, ogg):", type=["mp3", "wav", "ogg"])
    if st.button("Show Lyrics"):
        with open("temp_audio_file", "wb") as f:
            f.write(uploaded_file.getbuffer())
        result = transcript_model.transcribe("temp_audio_file")
        st.subheader("📝 Lyrics")
        st.write(result["text"])
elif sidebar_option == "HappyVibes":
    mood_requried=st.text_input("🔍 Enter the Mood of a song:")
    suggested_songs = songs_data[songs_data['Mood'] == mood_requried]
    st.write(suggested_songs[['song_title', 'artist']])


