import os.path
import random
import streamlit as st
import pandas as pd


m3u_filepaths_file = 'playlists/streamlit.m3u8'
ESSENTIA_ANALYSIS_PATH = 'data/descriptors.csv'


st.write('# Audio analysis playlists')
st.write(f'Using analysis data from `{ESSENTIA_ANALYSIS_PATH}`.')
audio_analysis = pd.read_csv(ESSENTIA_ANALYSIS_PATH, index_col=0)
audio_analysis_styles = audio_analysis.columns[:-5]
min_tempo = float(audio_analysis.min().tempo)
max_tempo = float(audio_analysis.max().tempo)
st.write('Loaded audio analysis for', len(audio_analysis), 'tracks.')

st.write('## 🔍 Select')
st.write('### By style')
st.write('Style activation statistics:')
st.write(audio_analysis.iloc[:, :-5].describe())

style_select = st.multiselect(
    'Select by style activations:', audio_analysis_styles)
if style_select:
    # Show the distribution of activation values for the selected styles.
    st.write(audio_analysis[style_select].describe())

    style_select_str = ', '.join(style_select)
    style_select_range = st.slider(
        f'Select tracks with `{style_select_str}` activations within range:', value=[0.5, 1.])

st.write('## 🔝 Rank Styles')
style_rank = st.multiselect(
    'Rank by style activations (multiplies activations for selected styles):', audio_analysis_styles, [])

st.write('## 🕘 Tempo')
tempo_select_range = st.slider(
    f'Select tracks with tempo within range:', min_value=min_tempo, max_value=max_tempo, value=[min_tempo, max_tempo])

st.write('## 🗣️ Vocals or Instrumental')
vocal_instr = st.radio(
    "Music with vocals or instrumental",
    ('Both', 'Vocals', 'Instrumental')
)

st.write('## 💃 Danceability')
danceability_select_range = st.slider(
    f'Select tracks with danceability within range:', value=[0., 1.])

st.write('## 🔥 Arousal')
arousal_select_range = st.slider(
    f'Select tracks with arousal within range:', min_value=1., max_value=9., value=[1., 9.])

st.write('## ☯️ Valence')
valence_select_range = st.slider(
    f'Select tracks with valence within range:', min_value=1., max_value=9., value=[1., 9.])

st.write('## 🔀 Post-process')
max_tracks = st.number_input('Maximum number of tracks (0 for all):', value=0)
shuffle = st.checkbox('Random shuffle')

if st.button("RUN"):
    st.write('## 🔊 Results')
    mp3s = list(audio_analysis.index)

    if style_select:
        audio_analysis_query = audio_analysis.loc[mp3s][style_select]

        result = audio_analysis_query
        for style in style_select:
            result = result.loc[result[style] >= style_select_range[0]]
        mp3s = result.index

    if tempo_select_range:
        audio_analysis_query = audio_analysis.loc[mp3s]
        result = audio_analysis_query
        result = result.loc[(result['tempo'] >= tempo_select_range[0])
                            & (result['tempo'] <= tempo_select_range[1])]
        mp3s = result.index

    if vocal_instr:
        audio_analysis_query = audio_analysis.loc[mp3s]
        result = audio_analysis_query

        if vocal_instr == 'Vocals':
            result = result.loc[result['voice'] >= 0.5]
        elif vocal_instr == 'Instrumental':
            result = result.loc[result['voice'] < 0.5]

        mp3s = result.index

    if danceability_select_range:
        audio_analysis_query = audio_analysis.loc[mp3s]
        result = audio_analysis_query
        result = result.loc[(result['danceability'] >= danceability_select_range[0])
                            & (result['danceability'] <= danceability_select_range[1])]
        mp3s = result.index

    if arousal_select_range:
        audio_analysis_query = audio_analysis.loc[mp3s]
        result = audio_analysis_query
        result = result.loc[(result['arousal'] >= arousal_select_range[0])
                            & (result['arousal'] <= arousal_select_range[1])]
        mp3s = result.index

    if valence_select_range:
        audio_analysis_query = audio_analysis.loc[mp3s]
        result = audio_analysis_query
        result = result.loc[(result['valence'] >= valence_select_range[0])
                            & (result['valence'] <= valence_select_range[1])]
        mp3s = result.index

    st.write(result)

    if style_rank:
        audio_analysis_query = audio_analysis.loc[mp3s][style_rank]
        audio_analysis_query['RANK'] = audio_analysis_query[style_rank[0]]
        for style in style_rank[1:]:
            audio_analysis_query['RANK'] *= audio_analysis_query[style]
        ranked = audio_analysis_query.sort_values(['RANK'], ascending=[False])
        ranked = ranked[['RANK'] + style_rank]
        mp3s = list(ranked.index)

        st.write('Applied ranking by audio style predictions.')
        st.write(ranked)

    if max_tracks:
        mp3s = mp3s[:max_tracks]
        st.write('Using top', len(mp3s), 'tracks from the results.')

    if shuffle:
        random.shuffle(mp3s)
        st.write('Applied random shuffle.')

    # Store the M3U8 playlist.
    with open(m3u_filepaths_file, 'w') as f:
        # Modify relative mp3 paths to make them accessible from the playlist folder.
        mp3_paths = [os.path.join('..', mp3) for mp3 in mp3s]
        f.write('\n'.join(mp3_paths))
        st.write(
            f'Stored M3U playlist (local filepaths) to `{m3u_filepaths_file}`.')

    st.write('Audio previews for the first 10 results:')
    for mp3 in mp3s[:10]:
        st.audio('audio/' + mp3, format="audio/mp3", start_time=0)
