# Audio Mood Classifier

Predicts the emotional mood of a song from the raw audio itself — no Spotify metadata required at inference time. Upload an MP3, get a mood, and optionally get the lyrics transcribed straight out of the waveform.

## Problem

Music mood classification is normally done with Spotify's precomputed audio features (`valence`, `energy`, `danceability`). That works right up until you have a track Spotify has never seen — a local file, an unreleased mix, a regional release outside their catalogue. Then you have nothing.

This project asks whether those features can be reconstructed from the audio signal directly, and whether a model trained on Spotify-labelled data still works when fed reconstructed features instead of official ones.

Two halves:

1. **Feature reconstruction** — derive Spotify-equivalent descriptors from raw audio with librosa.
2. **Mood classification** — train on a Spotify-labelled corpus, then predict on any audio file.

Plus lyric transcription via Whisper, because once you have the audio loaded, the words are a short step away.

## Feature reconstruction

`features.py` maps signal-processing primitives onto Spotify's feature vocabulary:

| Spotify feature | Reconstructed from |
|---|---|
| `tempo` | `librosa.beat.beat_track` |
| `loudness` | RMS energy mean |
| `key` | chroma CQT argmax |
| `mode` | chroma mean threshold |
| `energy` | mean squared amplitude |
| `speechiness` | spectral rolloff / sample rate |
| `acousticness` | spectral contrast mean |
| `instrumentalness` | harmonic component mean (HPSS) |
| `liveness` | spectral flatness |
| `danceability` | tempo-normalised + inverse zero-crossing rate |
| `duration_ms` | `librosa.get_duration` |

These are approximations, not reimplementations — Spotify's exact definitions are proprietary. The working assumption is that a classifier trained on Spotify features tolerates a monotonic approximation of them, and empirically it mostly does.

**`valence` is the exception and it is hardcoded.** Spotify's valence is itself a learned model output with no signal-processing analogue, so there is nothing to reconstruct it from. It sits at a `0.5` placeholder — an honest gap, and the single biggest source of error in the pipeline, since valence is the feature most directly about mood.

## Classification

- Labelled Spotify corpus → mood labels (`labeled_songs_with_moods.csv`)
- `StandardScaler` + supervised classifier, both persisted (`scaler.pkl`, `trained_mood_model.pkl`)
- Both a supervised and an unsupervised (clustering) approach were explored — the notebooks keep both so the comparison is visible

## Repository layout

```
features.py              # librosa feature reconstruction
spotify_prediction.py    # Streamlit app — upload audio, predict mood
newapp.py                # variant with Whisper lyric transcription
Streamlit_app.py         # Spotify search + mood lookup by track
transcript.py            # Whisper transcription to text file
new.py                   # feature extraction experiments
trained_mood_model.pkl   # trained classifier
scaler.pkl               # fitted scaler
labeled_songs_with_moods.csv
```

## Stack

Python · librosa · scikit-learn · OpenAI Whisper · Streamlit · Spotipy · pandas · joblib

## Setup

```bash
pip install -r requriments.txt
cp .env.example .env
streamlit run spotify_prediction.py
```

For the Spotify search app (`Streamlit_app.py`), register an app at the
[Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
and set:

| Variable | Purpose |
|---|---|
| `SPOTIFY_CLIENT_ID` | Spotify Web API |
| `SPOTIFY_CLIENT_SECRET` | Spotify Web API |

Credentials are read from the environment. `.env` is gitignored — do not commit keys.

## Known limitations

- **`valence` is a constant.** The feature that most directly encodes mood is the one that cannot be reconstructed from signal processing. Fixing it needs a separate valence regressor trained on Spotify labels.
- Reconstructed features are approximations of proprietary definitions, so there is unavoidable train/serve skew between the Spotify-labelled training data and librosa-derived inference features.
- Mood labels are coarse and culturally loaded — "happy" is not a property of a waveform.
- Whisper `base` is fast but weak on sung vocals with heavy instrumentation. `medium` or `large` transcribe lyrics far better at real cost.
