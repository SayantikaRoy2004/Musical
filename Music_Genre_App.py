# # # # import streamlit as st
# # # # import tensorflow as tf
# # # # import numpy as np
# # # # import librosa
# # # # import time
# # # # from tensorflow.image import resize
# # # # import random
# # # # import plotly.express as px
# # # # import pandas as pd
# # # #
# # # # # ------------------------------------
# # # # # Load model once and cache
# # # # # ------------------------------------
# # # # @st.cache_resource()
# # # # def load_model():
# # # #     return tf.keras.models.load_model("Trained_model.h5")
# # # #
# # # # # ------------------------------------
# # # # # Audio preprocessing
# # # # # ------------------------------------
# # # # def load_and_preprocess_data(file_obj, target_shape=(150, 150)):
# # # #     try:
# # # #         audio_data, sample_rate = librosa.load(file_obj, sr=None)
# # # #     except Exception as e:
# # # #         raise RuntimeError(f"Error loading audio: {e}")
# # # #
# # # #     audio_data = audio_data[:sample_rate * 30]
# # # #     chunk_duration = 4
# # # #     overlap_duration = 2
# # # #     chunk_samples = chunk_duration * sample_rate
# # # #     overlap_samples = overlap_duration * sample_rate
# # # #     num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
# # # #
# # # #     data = []
# # # #     for i in range(num_chunks):
# # # #         start = i * (chunk_samples - overlap_samples)
# # # #         end = start + chunk_samples
# # # #         chunk = audio_data[start:end]
# # # #         mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
# # # #         mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
# # # #         data.append(mel_spectrogram)
# # # #
# # # #     return np.array(data)
# # # #
# # # # # ------------------------------------
# # # # # Prediction logic
# # # # # ------------------------------------
# # # # def model_prediction(X_test):
# # # #     model = load_model()
# # # #     y_pred = model.predict(X_test)
# # # #     mean_pred = np.mean(y_pred, axis=0)
# # # #     predicted_index = np.argmax(mean_pred)
# # # #     return predicted_index, mean_pred
# # # #
# # # # # ------------------------------------
# # # # # Custom CSS – Dark Orange Spotify Style
# # # # # ------------------------------------
# # # # st.markdown("""
# # # # <style>
# # # # .stApp {
# # # #     background-color: #0e0e0e;
# # # #     color: #f5f5f5;
# # # #     font-family: 'Inter', sans-serif;
# # # # }
# # # # [data-testid="stSidebar"] {
# # # #     background-color: #141414;
# # # #     border-right: 1px solid #1f1f1f;
# # # # }
# # # # [data-testid="stSidebar"] * {
# # # #     color: #f1f1f1 !important;
# # # # }
# # # # .stButton>button {
# # # #     background-color: #ff8c00;
# # # #     color: #0f0f0f;
# # # #     border: none;
# # # #     border-radius: 10px;
# # # #     padding: 0.6rem 1.2rem;
# # # #     font-weight: 600;
# # # #     font-size: 1rem;
# # # #     transition: 0.25s ease-in-out;
# # # # }
# # # # .stButton>button:hover {
# # # #     background-color: #ffa733;
# # # #     transform: translateY(-2px);
# # # # }
# # # # .stFileUploader {
# # # #     background-color: #1a1a1a;
# # # #     border: 1px solid #2a2a2a;
# # # #     border-radius: 10px;
# # # #     padding: 1rem;
# # # # }
# # # # .info-card {
# # # #     background-color: #1a1a1a;
# # # #     border-radius: 12px;
# # # #     padding: 1.2rem 1.5rem;
# # # #     border: 1px solid #2a2a2a;
# # # #     margin-top: 1rem;
# # # # }
# # # # .conf-bar {
# # # #     background-color: #2a2a2a;
# # # #     height: 18px;
# # # #     border-radius: 6px;
# # # #     overflow: hidden;
# # # #     position: relative;
# # # #     margin-bottom: 8px;
# # # # }
# # # # .conf-fill {
# # # #     background-color: #ff8c00;
# # # #     height: 100%;
# # # #     border-radius: 6px 0 0 6px;
# # # # }
# # # # .conf-text {
# # # #     font-size: 0.9rem;
# # # #     margin-bottom: 4px;
# # # # }
# # # # .wave-container {
# # # #     display: flex;
# # # #     justify-content: center;
# # # #     align-items: flex-end;
# # # #     height: 80px;
# # # #     margin-top: 20px;
# # # # }
# # # # .wave-bar {
# # # #     width: 6px;
# # # #     margin: 0 3px;
# # # #     background-color: #ff8c00;
# # # #     border-radius: 3px;
# # # #     transition: height 0.15s ease-in-out;
# # # # }
# # # # </style>
# # # # """, unsafe_allow_html=True)
# # # #
# # # # # ------------------------------------
# # # # # Sidebar Navigation
# # # # # ------------------------------------
# # # # st.sidebar.title("Control Panel")
# # # # app_mode = st.sidebar.radio("Navigation", ["Home", "About", "Predict"])
# # # #
# # # # # ------------------------------------
# # # # # Home Page
# # # # # ------------------------------------
# # # # if app_mode == "Home":
# # # #     st.title("Music Genre Classification System")
# # # #     st.markdown("""
# # # #     **Welcome to the Music Genre Classification System**
# # # #     Upload a track and experience real-time deep learning genre prediction with an interactive player and confidence visualization.
# # # #
# # # #     ---
# # # #     **Features**
# # # #     - Confidence breakdown for all genres
# # # #     - Deep CNN trained on GTZAN dataset
# # # #     """)
# # # #
# # # # # ------------------------------------
# # # # # About Page
# # # # # ------------------------------------
# # # # elif app_mode == "About":
# # # #     st.title("About This Project")
# # # #     st.markdown("""
# # # #     This system uses **Convolutional Neural Networks (CNNs)** trained on **Mel-Spectrograms** to detect audio genres.
# # # #
# # # #     **Dataset:** GTZAN (10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
# # # #     Each file is analyzed for spectral and rhythmic characteristics, converted into a mel-spectrogram, and classified using a deep learning model.
# # # #     """)
# # # #
# # # # # ------------------------------------
# # # # # Prediction Page - Upgraded UI
# # # # # ------------------------------------
# # # # # ------------------------------------
# # # # # Prediction Page - Clean & Polished
# # # # # ------------------------------------
# # # # elif app_mode == "Predict":
# # # #     st.title("🎵 Music Genre Prediction")
# # # #     uploaded_file = st.file_uploader("Upload MP3 or WAV file", type=["mp3", "wav"])
# # # #
# # # #     if uploaded_file is not None:
# # # #         st.markdown("---")
# # # #         # Two-column layout: audio + waveform | predictions
# # # #         col_audio, col_results = st.columns([1, 1])
# # # #
# # # #         # ---------------- Audio & Waveform ----------------
# # # #         with col_audio:
# # # #             st.subheader("🎧 Audio Player")
# # # #             st.audio(uploaded_file)
# # # #
# # # #             st.subheader("🔊 Audio Visualizer")
# # # #             placeholder = st.empty()
# # # #             num_bars = 20
# # # #
# # # #             if st.button("Play & Visualize Audio"):
# # # #                 # Simulated waveform
# # # #                 for _ in range(20):
# # # #                     bars = [random.randint(10, 70) for _ in range(num_bars)]
# # # #                     html_bars = "".join([f'<div class="wave-bar" style="height:{b}px;"></div>' for b in bars])
# # # #                     placeholder.markdown(f'<div class="wave-container">{html_bars}</div>', unsafe_allow_html=True)
# # # #                     time.sleep(0.1)
# # # #
# # # #         # ---------------- Prediction Results ----------------
# # # #         with col_results:
# # # #             st.subheader("🎯 Prediction")
# # # #             if st.button("Predict Genre"):
# # # #                 with st.spinner("Analyzing your audio..."):
# # # #                     try:
# # # #                         start = time.time()
# # # #                         X_test = load_and_preprocess_data(uploaded_file)
# # # #                         pred_idx, mean_pred = model_prediction(X_test)
# # # #                         duration = time.time() - start
# # # #
# # # #                         labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
# # # #                                   'jazz', 'metal', 'pop', 'reggae', 'rock']
# # # #                         confidence = (mean_pred / np.sum(mean_pred)) * 100
# # # #
# # # #                         # ----- Predicted Genre Card -----
# # # #                         st.markdown(f"""
# # # #                         <div class="info-card">
# # # #                             <h2>Predicted Genre: <span style="color:#ff8c00">{labels[pred_idx].capitalize()}</span></h2>
# # # #                             <p>Processing Time: {duration:.2f} seconds</p>
# # # #                         </div>
# # # #                         """, unsafe_allow_html=True)
# # # #
# # # #                         # ----- Top 3 Predictions -----
# # # #                         st.markdown("### 🏆 Top 3 Predictions")
# # # #                         top_idx = np.argsort(confidence)[::-1][:3]
# # # #                         for i in top_idx:
# # # #                             st.markdown(f"**{labels[i].capitalize()}** — {confidence[i]:.2f}%")
# # # #
# # # #                         # ----- Confidence Chart -----
# # # #                         st.markdown("### 📊 Confidence Breakdown")
# # # #                         df = pd.DataFrame({
# # # #                             'Genre': labels,
# # # #                             'Confidence (%)': confidence
# # # #                         })
# # # #                         fig = px.bar(df, x='Genre', y='Confidence (%)', color='Confidence (%)',
# # # #                                      color_continuous_scale='Oranges', text='Confidence (%)')
# # # #                         fig.update_layout(showlegend=False, yaxis_range=[0, 100], title=None)
# # # #                         st.plotly_chart(fig, use_container_width=True)
# # # #
# # # #                     except Exception as e:
# # # #                         st.error(f"Prediction failed: {e}")
# # # #
# # # import streamlit as st
# # # import tensorflow as tf
# # # import numpy as np
# # # import librosa
# # # import time
# # # from tensorflow.image import resize
# # # import plotly.express as px
# # # import pandas as pd
# # #
# # # # ------------------------------------
# # # # Load model once and cache
# # # # ------------------------------------
# # # @st.cache_resource()
# # # def load_model():
# # #     return tf.keras.models.load_model("Trained_model.h5")
# # #
# # # # ------------------------------------
# # # # Audio preprocessing
# # # # ------------------------------------
# # # def load_and_preprocess_data(file_obj, target_shape=(150, 150)):
# # #     try:
# # #         audio_data, sample_rate = librosa.load(file_obj, sr=None)
# # #     except Exception as e:
# # #         raise RuntimeError(f"Error loading audio: {e}")
# # #
# # #     audio_data = audio_data[:sample_rate * 30]
# # #     chunk_duration = 4
# # #     overlap_duration = 2
# # #     chunk_samples = chunk_duration * sample_rate
# # #     overlap_samples = overlap_duration * sample_rate
# # #     num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
# # #
# # #     data = []
# # #     for i in range(num_chunks):
# # #         start = i * (chunk_samples - overlap_samples)
# # #         end = start + chunk_samples
# # #         chunk = audio_data[start:end]
# # #         mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
# # #         mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
# # #         data.append(mel_spectrogram)
# # #
# # #     return np.array(data)
# # #
# # # # ------------------------------------
# # # # Prediction logic
# # # # ------------------------------------
# # # def model_prediction(X_test):
# # #     model = load_model()
# # #     y_pred = model.predict(X_test)
# # #     mean_pred = np.mean(y_pred, axis=0)
# # #     predicted_index = np.argmax(mean_pred)
# # #     return predicted_index, mean_pred
# # #
# # # # ------------------------------------
# # # # Custom CSS
# # # # ------------------------------------
# # # st.markdown("""
# # # <style>
# # # .stApp {background-color: #0e0e0e; color: #f5f5f5; font-family: 'Inter', sans-serif;}
# # # [data-testid="stSidebar"] {background-color: #141414; border-right: 1px solid #1f1f1f;}
# # # [data-testid="stSidebar"] * {color: #f1f1f1 !important;}
# # # .stButton>button {background-color: #ff8c00; color: #0f0f0f; border: none; border-radius: 10px; padding: 0.6rem 1.2rem; font-weight: 600; font-size: 1rem; transition: 0.25s ease-in-out;}
# # # .stButton>button:hover {background-color: #ffa733; transform: translateY(-2px);}
# # # .stFileUploader {background-color: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px; padding: 1rem;}
# # # .info-card {background-color: #1a1a1a; border-radius: 12px; padding: 1.2rem 1.5rem; border: 1px solid #2a2a2a; margin-top: 1rem;}
# # # .wave-container {display: flex; justify-content: center; align-items: flex-end; height: 80px; margin-top: 20px;}
# # # .wave-bar {width: 6px; margin: 0 3px; background-color: #ff8c00; border-radius: 3px; transition: height 0.1s ease-in-out;}
# # # </style>
# # # """, unsafe_allow_html=True)
# # #
# # # # ------------------------------------
# # # # Sidebar Navigation
# # # # ------------------------------------
# # # st.sidebar.title("Control Panel")
# # # app_mode = st.sidebar.radio("Navigation", ["Home", "About", "Predict"])
# # #
# # # # ------------------------------------
# # # # Home Page
# # # # ------------------------------------
# # # if app_mode == "Home":
# # #     st.title("Music Genre Classification System")
# # #     st.markdown("""
# # #     **Welcome to the Music Genre Classification System**
# # #     Upload a track and experience real-time deep learning genre prediction with an interactive player and confidence visualization.
# # #
# # #     ---
# # #     **Features**
# # #     - Real-time waveform visualization
# # #     - Confidence breakdown for all genres
# # #     - Deep CNN trained on GTZAN dataset
# # #     """)
# # #
# # # # ------------------------------------
# # # # About Page
# # # # ------------------------------------
# # # elif app_mode == "About":
# # #     st.title("About This Project")
# # #     st.markdown("""
# # #     This system uses **Convolutional Neural Networks (CNNs)** trained on **Mel-Spectrograms** to detect audio genres.
# # #
# # #     **Dataset:** GTZAN (10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
# # #     Each file is analyzed for spectral and rhythmic characteristics, converted into a mel-spectrogram, and classified using a deep learning model.
# # #     """)
# # #
# # # # ------------------------------------
# # # # Prediction Page
# # # # ------------------------------------
# # # elif app_mode == "Predict":
# # #     st.title("🎵 Music Genre Prediction")
# # #     uploaded_file = st.file_uploader("Upload MP3 or WAV file", type=["mp3", "wav"])
# # #
# # #     if uploaded_file is not None:
# # #         st.markdown("---")
# # #         col_audio, col_results = st.columns([1, 1])
# # #
# # #         # ---------------- Audio & Waveform ----------------
# # #         with col_audio:
# # #             st.subheader("🎧 Audio Player")
# # #             st.audio(uploaded_file)
# # #
# # #             st.subheader("🔊 Audio Visualizer")
# # #             placeholder = st.empty()
# # #             num_bars = 30
# # #
# # #             if st.button("Play & Visualize Audio"):
# # #                 audio_data, sr = librosa.load(uploaded_file, sr=None)
# # #                 audio_data = audio_data[:sr*30]  # max 30 sec
# # #                 audio_data = audio_data / np.max(np.abs(audio_data))
# # #                 chunk_size = len(audio_data) // num_bars
# # #                 bars_height = [np.mean(np.abs(audio_data[i*chunk_size:(i+1)*chunk_size]))*70 for i in range(num_bars)]
# # #
# # #                 for _ in range(40):
# # #                     html_bars = "".join([f'<div class="wave-bar" style="height:{int(h)}px;"></div>' for h in bars_height])
# # #                     placeholder.markdown(f'<div class="wave-container">{html_bars}</div>', unsafe_allow_html=True)
# # #                     bars_height = bars_height[1:] + bars_height[:1]  # shift to simulate movement
# # #                     time.sleep(0.08)
# # #
# # #         # ---------------- Prediction Results ----------------
# # #         with col_results:
# # #             st.subheader("🎯 Prediction")
# # #             if st.button("Predict Genre"):
# # #                 with st.spinner("Analyzing your audio..."):
# # #                     try:
# # #                         start = time.time()
# # #                         X_test = load_and_preprocess_data(uploaded_file)
# # #                         pred_idx, mean_pred = model_prediction(X_test)
# # #                         duration = time.time() - start
# # #
# # #                         labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
# # #                                   'jazz', 'metal', 'pop', 'reggae', 'rock']
# # #                         confidence = (mean_pred / np.sum(mean_pred)) * 100
# # #
# # #                         # Predicted Genre Card
# # #                         st.markdown(f"""
# # #                         <div class="info-card">
# # #                             <h2>Predicted Genre: <span style="color:#ff8c00">{labels[pred_idx].capitalize()}</span></h2>
# # #                             <p>Processing Time: {duration:.2f} seconds</p>
# # #                         </div>
# # #                         """, unsafe_allow_html=True)
# # #
# # #                         # Top 3 Predictions
# # #                         st.markdown("### 🏆 Top 3 Predictions")
# # #                         top_idx = np.argsort(confidence)[::-1][:3]
# # #                         for i in top_idx:
# # #                             st.markdown(f"**{labels[i].capitalize()}** — {confidence[i]:.2f}%")
# # #
# # #                         # Confidence Chart
# # #                         st.markdown("### 📊 Confidence Breakdown")
# # #                         df = pd.DataFrame({'Genre': labels, 'Confidence (%)': confidence})
# # #                         fig = px.bar(df, x='Genre', y='Confidence (%)', color='Confidence (%)',
# # #                                      color_continuous_scale='Oranges', text='Confidence (%)')
# # #                         fig.update_layout(showlegend=False, yaxis_range=[0, 100], title=None)
# # #                         st.plotly_chart(fig, use_container_width=True)
# # #
# # #                     except Exception as e:
# # #                         st.error(f"Prediction failed: {e}")
# # import streamlit as st
# # import tensorflow as tf
# # import numpy as np
# # import librosa
# # import time
# # from tensorflow.image import resize
# # import plotly.express as px
# # import pandas as pd
# #
# # # ---------------- Load model ----------------
# # @st.cache_resource()
# # def load_model():
# #     return tf.keras.models.load_model("Trained_model.h5")
# #
# # # ---------------- Preprocess audio ----------------
# # def load_and_preprocess_data(file_obj, target_shape=(150, 150)):
# #     audio_data, sample_rate = librosa.load(file_obj, sr=None)
# #     audio_data = audio_data[:sample_rate * 30]
# #     chunk_duration = 4
# #     overlap_duration = 2
# #     chunk_samples = chunk_duration * sample_rate
# #     overlap_samples = overlap_duration * sample_rate
# #     num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
# #
# #     data = []
# #     for i in range(num_chunks):
# #         start = i * (chunk_samples - overlap_samples)
# #         end = start + chunk_samples
# #         chunk = audio_data[start:end]
# #         mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
# #         mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
# #         data.append(mel_spectrogram)
# #     return np.array(data)
# #
# # # ---------------- Prediction ----------------
# # def model_prediction(X_test):
# #     model = load_model()
# #     y_pred = model.predict(X_test)
# #     mean_pred = np.mean(y_pred, axis=0)
# #     predicted_index = np.argmax(mean_pred)
# #     return predicted_index, mean_pred
# #
# # # ---------------- CSS ----------------
# # st.markdown("""
# # <style>
# # .stApp {background-color: #0e0e0e; color: #f5f5f5; font-family: 'Inter', sans-serif;}
# # [data-testid="stSidebar"] {background-color: #141414; border-right: 1px solid #1f1f1f;}
# # [data-testid="stSidebar"] * {color: #f1f1f1 !important;}
# # .stButton>button {background-color: #ff8c00; color: #0f0f0f; border: none; border-radius: 10px; padding: 0.6rem 1.2rem; font-weight: 600; font-size: 1rem; transition: 0.25s ease-in-out;}
# # .stButton>button:hover {background-color: #ffa733; transform: translateY(-2px);}
# # .stFileUploader {background-color: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px; padding: 1rem;}
# # .info-card {background-color: #1a1a1a; border-radius: 12px; padding: 1.2rem 1.5rem; border: 1px solid #2a2a2a; margin-top: 1rem;}
# # .wave-container {display: flex; justify-content: center; align-items: flex-end; height: 120px; margin-top: 20px;}
# # .wave-bar {width: 4px; margin: 0 2px; border-radius: 3px; transition: height 0.05s ease-in-out;}
# # </style>
# # """, unsafe_allow_html=True)
# #
# # # ---------------- Sidebar ----------------
# # st.sidebar.title("Control Panel")
# # app_mode = st.sidebar.radio("Navigation", ["Home", "About", "Predict"])
# #
# # # ---------------- Home ----------------
# # if app_mode == "Home":
# #     st.title("🎶 Supreme Music Genre Classifier")
# #     st.markdown("""
# #     **Experience real-time, ultra-responsive music visualization & genre prediction**
# #     Upload your track and watch it come alive with a professional-grade visualizer.
# #
# #     ---
# #     **Features**:
# #     - Supreme live waveform visualizer
# #     - Neon-style gradient bars with stereo mirroring
# #     - Deep CNN genre predictions with confidence chart
# #     """)
# #
# # # ---------------- About ----------------
# # elif app_mode == "About":
# #     st.title("About This Project")
# #     st.markdown("""
# #     Uses **CNN trained on Mel-Spectrograms** to classify music genres.
# #
# #     **Dataset:** GTZAN (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
# #     Audio is analyzed for spectral & rhythmic features, converted into Mel-spectrograms, and predicted using a deep model.
# #     """)
# #
# # # ---------------- Predict ----------------
# # elif app_mode == "Predict":
# #     st.title("🎵 Music Genre Prediction")
# #     uploaded_file = st.file_uploader("Upload MP3 or WAV", type=["mp3", "wav"])
# #
# #     if uploaded_file is not None:
# #         st.markdown("---")
# #         col_audio, col_results = st.columns([1, 1])
# #
# #         # ---- Audio Player + Supreme Visualizer ----
# #         with col_audio:
# #             st.subheader("🎧 Audio Player")
# #             st.audio(uploaded_file)
# #
# #             st.subheader("🌟 Supreme Visualizer")
# #             placeholder = st.empty()
# #             num_bars = 60
# #
# #             if st.button("Play & Visualize"):
# #                 audio_data, sr = librosa.load(uploaded_file, sr=None)
# #                 audio_data = audio_data[:sr*30]
# #                 audio_data = audio_data / np.max(np.abs(audio_data))
# #
# #                 chunk_size = len(audio_data) // num_bars
# #                 bars = [np.mean(np.abs(audio_data[i*chunk_size:(i+1)*chunk_size]))*80 for i in range(num_bars)]
# #
# #                 for _ in range(150):  # frames
# #                     # smooth breathing + pulsing
# #                     bars = [np.clip(b + np.sin(time.time()*4 + i)*8, 5, 120) for i, b in enumerate(bars)]
# #                     html_bars = "".join([
# #                         f'<div class="wave-bar" style="height:{int(h)}px; background: linear-gradient(to top, #ff00ff, #ff8c00);"></div>'
# #                         for h in bars
# #                     ])
# #                     # mirrored effect
# #                     html_bars_mirror = html_bars[::-1]
# #                     placeholder.markdown(f'<div class="wave-container">{html_bars}{html_bars_mirror}</div>', unsafe_allow_html=True)
# #
# #                     bars = bars[1:] + bars[:1]  # shift for flowing effect
# #                     time.sleep(0.05)
# #
# #         # ---- Predictions ----
# #         with col_results:
# #             st.subheader("🎯 Genre Prediction")
# #             if st.button("Predict Genre"):
# #                 with st.spinner("Analyzing audio..."):
# #                     try:
# #                         start = time.time()
# #                         X_test = load_and_preprocess_data(uploaded_file)
# #                         pred_idx, mean_pred = model_prediction(X_test)
# #                         duration = time.time() - start
# #
# #                         labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
# #                                   'jazz', 'metal', 'pop', 'reggae', 'rock']
# #                         confidence = (mean_pred / np.sum(mean_pred)) * 100
# #
# #                         # Predicted Genre Card
# #                         st.markdown(f"""
# #                         <div class="info-card">
# #                             <h2>Predicted Genre: <span style="color:#ff00ff">{labels[pred_idx].capitalize()}</span></h2>
# #                             <p>Processing Time: {duration:.2f} seconds</p>
# #                         </div>
# #                         """, unsafe_allow_html=True)
# #
# #                         # Top 3 Predictions
# #                         st.markdown("### 🏆 Top 3 Predictions")
# #                         top_idx = np.argsort(confidence)[::-1][:3]
# #                         for i in top_idx:
# #                             st.markdown(f"**{labels[i].capitalize()}** — {confidence[i]:.2f}%")
# #
# #                         # Confidence Chart
# #                         st.markdown("### 📊 Confidence Breakdown")
# #                         df = pd.DataFrame({'Genre': labels, 'Confidence (%)': confidence})
# #                         fig = px.bar(df, x='Genre', y='Confidence (%)', color='Confidence (%)',
# #                                      color_continuous_scale='Plasma', text='Confidence (%)')
# #                         fig.update_layout(showlegend=False, yaxis_range=[0, 100], title=None)
# #                         st.plotly_chart(fig, use_container_width=True)
# #
# #                     except Exception as e:
# #                         st.error(f"Prediction failed: {e}")
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import librosa
# import time
# from tensorflow.image import resize
# import plotly.express as px
# import pandas as pd
#
# # ---------------------- Load Model ----------------------
# @st.cache_resource()
# def load_model():
#     return tf.keras.models.load_model("Trained_model.h5")
#
# # ---------------------- Audio Preprocessing ----------------------
# def load_and_preprocess_data(file_obj, target_shape=(150, 150)):
#     try:
#         audio_data, sample_rate = librosa.load(file_obj, sr=None)
#     except Exception as e:
#         raise RuntimeError(f"Error loading audio: {e}")
#
#     audio_data = audio_data[:sample_rate * 30]  # first 30s
#     chunk_duration = 4
#     overlap_duration = 2
#     chunk_samples = chunk_duration * sample_rate
#     overlap_samples = overlap_duration * sample_rate
#     num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
#
#     data = []
#     for i in range(num_chunks):
#         start = i * (chunk_samples - overlap_samples)
#         end = start + chunk_samples
#         chunk = audio_data[start:end]
#         mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
#         mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
#         data.append(mel_spectrogram)
#
#     return np.array(data)
#
# # ---------------------- Prediction Logic ----------------------
# def model_prediction(X_test):
#     model = load_model()
#     y_pred = model.predict(X_test)
#     mean_pred = np.mean(y_pred, axis=0)
#     predicted_index = np.argmax(mean_pred)
#     return predicted_index, mean_pred
#
# # ---------------------- Custom CSS ----------------------
# st.markdown("""
# <style>
# .stApp {background-color: #0e0e0e; color: #f5f5f5; font-family: 'Inter', sans-serif;}
# [data-testid="stSidebar"] {background-color: #141414; border-right: 1px solid #1f1f1f;}
# [data-testid="stSidebar"] * {color: #f1f1f1 !important;}
# .stButton>button {background-color: #ff8c00; color: #0f0f0f; border: none; border-radius: 10px; padding: 0.6rem 1.2rem; font-weight: 600; font-size: 1rem; transition: 0.25s ease-in-out;}
# .stButton>button:hover {background-color: #ffa733; transform: translateY(-2px);}
# .stFileUploader {background-color: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px; padding: 1rem;}
# .info-card {background-color: #1a1a1a; border-radius: 12px; padding: 1.2rem 1.5rem; border: 1px solid #2a2a2a; margin-top: 1rem;}
# .wave-container {display: flex; justify-content: center; align-items: flex-end; height: 100px; margin-top: 20px;}
# .wave-bar {width: 6px; margin: 0 2px; border-radius: 3px; transition: height 0.05s ease-in-out;}
# </style>
# """, unsafe_allow_html=True)
#
# # ---------------------- Sidebar ----------------------
# st.sidebar.title("Control Panel")
# app_mode = st.sidebar.radio("Navigation", ["Home", "About", "Predict"])
#
# # ---------------------- Home ----------------------
# if app_mode == "Home":
#     st.title("🎶 Music Genre Classification System")
#     st.markdown("""
#     **Welcome!** Upload a music track and see its genre predicted in real-time with a **living visualizer**.
#
#     ---
#     **Features:**
#     - Real-time audio visualization
#     - Deep CNN trained on GTZAN dataset
#     - Top 3 predictions + confidence chart
#     """)
#
# # ---------------------- About ----------------------
# elif app_mode == "About":
#     st.title("About This Project")
#     st.markdown("""
#     This system uses **Convolutional Neural Networks (CNNs)** trained on **Mel-Spectrograms** to classify audio genres.
#
#     **Dataset:** GTZAN (10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
#
#     Each audio file is analyzed for spectral and rhythmic characteristics and classified with deep learning.
#     """)
#
# # ---------------------- Predict ----------------------
# elif app_mode == "Predict":
#     st.title("🎵 Supreme Music Genre Prediction")
#     uploaded_file = st.file_uploader("Upload MP3 or WAV file", type=["mp3", "wav"])
#
#     if uploaded_file is not None:
#         st.markdown("---")
#         col_audio, col_results = st.columns([1,1])
#
#         # ---------------- Audio Player + Supreme Visualizer ----------------
#         with col_audio:
#             st.subheader("🎧 Audio Player")
#             st.audio(uploaded_file)
#
#             st.subheader("🌟 Supreme Visualizer")
#             placeholder = st.empty()
#             num_bars = 60
#
#             # Load audio waveform
#             audio_data, sr = librosa.load(uploaded_file, sr=None)
#             audio_data = audio_data[:sr*30]  # first 30 seconds
#             audio_data = audio_data / np.max(np.abs(audio_data))  # normalize
#
#             chunk_size = len(audio_data) // num_bars
#             bars = [np.mean(np.abs(audio_data[i*chunk_size:(i+1)*chunk_size]))*100 for i in range(num_bars)]
#
#             # Animate waveform
#             for _ in range(200):
#                 bars = [np.clip(b + np.sin(time.time()*5 + i)*5, 5, 120) for i, b in enumerate(bars)]
#                 html_bars = "".join([f'<div class="wave-bar" style="height:{int(h)}px; background: linear-gradient(to top, #ff00ff, #ff8c00);"></div>' for h in bars])
#                 placeholder.markdown(f'<div class="wave-container">{html_bars}</div>', unsafe_allow_html=True)
#                 bars = bars[1:] + bars[:1]
#                 time.sleep(0.05)
#
#         # ---------------- Prediction Results ----------------
#         with col_results:
#             st.subheader("🎯 Prediction")
#             try:
#                 start = time.time()
#                 X_test = load_and_preprocess_data(uploaded_file)
#                 pred_idx, mean_pred = model_prediction(X_test)
#                 duration = time.time() - start
#
#                 labels = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
#                 confidence = (mean_pred / np.sum(mean_pred))*100
#
#             # Predicted Genre Card
#             #                         st.markdown(f"""
#             #                         <div class="info-card">
#             #                             <h2>Predicted Genre: <span style="color:#ff00ff">{labels[pred_idx].capitalize()}</span></h2>
#             #                             <p>Processing Time: {duration:.2f} seconds</p>
#             #                         </div>
#             #                         """, unsafe_allow_html=True)
#             #
#             #                         # Top 3 Predictions
#             #                         st.markdown("### 🏆 Top 3 Predictions")
#             #                         top_idx = np.argsort(confidence)[::-1][:3]
#             #                         for i in top_idx:
#             #                             st.markdown(f"**{labels[i].capitalize()}** — {confidence[i]:.2f}%")
#             #
#             #                         # Confidence Chart
#             #                         st.markdown("### 📊 Confidence Breakdown")
#             #                         df = pd.DataFrame({'Genre': labels, 'Confidence (%)': confidence})
#             #                         fig = px.bar(df, x='Genre', y='Confidence (%)', color='Confidence (%)',
#             #                                      color_continuous_scale='Plasma', text='Confidence (%)')
#             #                         fig.update_layout(showlegend=False, yaxis_range=[0, 100], title=None)
#             #                         st.plotly_chart(fig, use_container_width=True)
#             #
#             #                     except Exception as e:
#             #                         st.error(f"Prediction failed: {e}")
#
import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import time
from tensorflow.image import resize
import plotly.express as px
import pandas as pd
import tempfile

# ---------------------- Load Model ----------------------
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("Trained_model.h5")

# ---------------------- Audio Preprocessing ----------------------
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
    except Exception as e:
        raise RuntimeError(f"Error loading audio: {e}")

    audio_data = audio_data[:sample_rate * 30]  # first 30s
    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    data = []
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)

# ---------------------- Prediction Logic ----------------------
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    mean_pred = np.mean(y_pred, axis=0)
    predicted_index = np.argmax(mean_pred)
    return predicted_index, mean_pred

# ---------------------- Custom CSS ----------------------
st.markdown("""
<style>
.stApp {background-color: #0e0e0e; color: #f5f5f5; font-family: 'Inter', sans-serif;}
[data-testid="stSidebar"] {background-color: #141414; border-right: 1px solid #1f1f1f;}
[data-testid="stSidebar"] * {color: #f1f1f1 !important;}
.stButton>button {background-color: #ff8c00; color: #0f0f0f; border: none; border-radius: 10px; padding: 0.6rem 1.2rem; font-weight: 600; font-size: 1rem; transition: 0.25s ease-in-out;}
.stButton>button:hover {background-color: #ffa733; transform: translateY(-2px);}
.stFileUploader {background-color: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px; padding: 1rem;}
.info-card {background-color: #1a1a1a; border-radius: 12px; padding: 1.2rem 1.5rem; border: 1px solid #2a2a2a; margin-top: 1rem;}
.wave-container {display: flex; justify-content: center; align-items: flex-end; height: 100px; margin-top: 20px;}
.wave-bar {width: 6px; margin: 0 2px; border-radius: 3px; transition: height 0.05s ease-in-out;}
</style>
""", unsafe_allow_html=True)

# ---------------------- Sidebar ----------------------
st.sidebar.title("Control Panel")
app_mode = st.sidebar.radio("Navigation", ["Home", "About", "Predict"])

# ---------------------- Home ----------------------
if app_mode == "Home":
    st.title("🎶 Music Genre Classification System")
    st.markdown("""
    **Welcome!** Upload a music track and see its genre predicted in real-time with a **living visualizer**.

    ---
    **Features:**  
    - Real-time audio visualization  
    - Deep CNN trained on GTZAN dataset  
    - Top 3 predictions + confidence chart  
    """)

# ---------------------- About ----------------------
elif app_mode == "About":
    st.title("About This Project")
    st.markdown("""
    This system uses **Convolutional Neural Networks (CNNs)** trained on **Mel-Spectrograms** to classify audio genres.

    **Dataset:** GTZAN (10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)  

    Each audio file is analyzed for spectral and rhythmic characteristics and classified with deep learning.
    """)

# ---------------------- Predict ----------------------
elif app_mode == "Predict":
    st.title("🎵 Supreme Music Genre Prediction")
    uploaded_file = st.file_uploader("Upload MP3 or WAV file", type=["mp3", "wav"])

    if uploaded_file is not None:
        import tempfile

        # Save uploaded file to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_filename = tmp_file.name

        st.markdown("---")
        col_audio, col_results = st.columns([1,1])

        # Initialize session state for controlling animation
        if "animate" not in st.session_state:
            st.session_state.animate = False

        # ---------------- Audio Player + Supreme Visualizer ----------------
        with col_audio:
            st.subheader("🎧 Audio Player")
            st.audio(temp_filename)

            st.subheader("🌟 Supreme Visualizer")
            placeholder = st.empty()
            num_bars = 60

            # Load audio waveform
            audio_data, sr = librosa.load(temp_filename, sr=None)
            audio_data = audio_data[:sr*30]  # first 30 seconds
            audio_data = audio_data / np.max(np.abs(audio_data))  # normalize

            chunk_size = len(audio_data) // num_bars
            bars = [np.mean(np.abs(audio_data[i*chunk_size:(i+1)*chunk_size]))*100 for i in range(num_bars)]

            # Buttons to control animation
            col_play, col_stop = st.columns(2)
            if col_play.button("▶️ Play Visualizer"):
                st.session_state.animate = True
            if col_stop.button("⏸ Stop Visualizer"):
                st.session_state.animate = False

            # Animate while 'animate' is True
            while st.session_state.animate:
                bars = [np.clip(b + np.sin(time.time()*5 + i)*5, 5, 120) for i, b in enumerate(bars)]
                html_bars = "".join([f'<div class="wave-bar" style="height:{int(h)}px; background: linear-gradient(to top, #ff00ff, #ff8c00);"></div>' for h in bars])
                placeholder.markdown(f'<div class="wave-container">{html_bars}</div>', unsafe_allow_html=True)
                bars = bars[1:] + bars[:1]
                time.sleep(0.05)

        # ---------------- Prediction Results ----------------
        with col_results:
            st.subheader("🎯 Prediction")
            try:
                start = time.time()
                X_test = load_and_preprocess_data(temp_filename)
                pred_idx, mean_pred = model_prediction(X_test)
                duration = time.time() - start

                labels = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
                confidence = (mean_pred / np.sum(mean_pred))*100

                # Predicted Genre Card
                st.markdown(f"""
                <div class="info-card">
                    <h2>Predicted Genre: <span style="color:#ff00ff">{labels[pred_idx].capitalize()}</span></h2>
                    <p>Processing Time: {duration:.2f} seconds</p>
                </div>
                """, unsafe_allow_html=True)

                # Top 3 Predictions
                st.markdown("### 🏆 Top 3 Predictions")
                top_idx = np.argsort(confidence)[::-1][:3]
                for i in top_idx:
                    st.markdown(f"**{labels[i].capitalize()}** — {confidence[i]:.2f}%")

                # Confidence Chart
                st.markdown("### 📊 Confidence Breakdown")
                df = pd.DataFrame({'Genre': labels, 'Confidence (%)': confidence})
                fig = px.bar(df, x='Genre', y='Confidence (%)', color='Confidence (%)',
                             color_continuous_scale='Plasma', text='Confidence (%)')
                fig.update_layout(showlegend=False, yaxis_range=[0, 100], title=None)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
