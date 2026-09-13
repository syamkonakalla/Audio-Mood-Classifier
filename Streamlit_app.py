import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotify_prediction import predict_mood  # Import the function from prediction.py

# Spotify API credentials
import os
from dotenv import load_dotenv

load_dotenv()
cid = os.environ["SPOTIFY_CLIENT_ID"]
secret = os.environ["SPOTIFY_CLIENT_SECRET"]
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def search_tracks(query, limit=10):
    """Search for tracks using a query and return a list of track names, URIs, and images."""
    results = sp.search(q=query, type='track', limit=limit)
    tracks = results['tracks']['items']
    track_options = [(track['name'], track['uri'], track['album']['images'][0]['url']) for track in tracks]
    return track_options

# Streamlit UI
st.title("Spotify Mood Predictor")
st.write("Search for a song and select a track to predict the mood of the song.")

# Search box for the user to search for songs
search_query = st.text_input("Search for a song", "")

if search_query:
    # Get the search results
    track_options = search_tracks(search_query)

    if track_options:
        # Create a dropdown to select a song
        track_name, track_uri, track_image = zip(*track_options)  # Unzip the list of (name, URI, image URL) tuples
        selected_track = st.selectbox("Select a track", track_name)

        # Find the corresponding track URI and image URL for the selected song
        selected_track_uri = track_options[track_name.index(selected_track)][1]
        selected_track_image = track_options[track_name.index(selected_track)][2]

        # Display the album image
        st.image(selected_track_image, caption=f"Album Art: {selected_track}", use_column_width=True)

        # Add Spotify player iframe to play the track
        st.markdown(f'<iframe src="https://open.spotify.com/embed/track/{selected_track_uri.split(":")[-1]}" width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', unsafe_allow_html=True)

        # When the user selects a track, show the prediction button
        if st.button("Predict Mood"):
            # Predict the mood for the selected track
            mood = predict_mood(selected_track_uri)
            st.write(f"The predicted mood for '{selected_track}' is: {mood}")
    else:
        st.write("No tracks found. Try refining your search.")
else:
    st.write("Enter a song name to search for tracks.")

