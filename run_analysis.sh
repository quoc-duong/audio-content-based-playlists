#!/bin/bash
cd "$(dirname "$0")"

mkdir models
cd models

# Tempo model
wget -nc https://essentia.upf.edu/models/tempo/tempocnn/deepsquare-k16-3.pb

# Music styles model
wget -nc https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.pb

# Voice/instrument model
wget -nc https://essentia.upf.edu/models/classifiers/voice_instrumental/voice_instrumental-musicnn-msd-2.pb

# Danceability model
wget -nc https://essentia.upf.edu/models/classifiers/danceability/danceability-musicnn-msd-2.pb

# Embeddings and regression models + metadata
wget -nc https://essentia.upf.edu/models/autotagging/msd/msd-musicnn-1.pb
wget -nc https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-musicnn-msd-2.json
wget -nc https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-musicnn-msd-2.pb

cd ../analysis

# File containing Discogs labels
wget -nc https://raw.githubusercontent.com/MTG/essentia-replicate-demos/main/effnet-discogs/labels.py

python3 analysis.py