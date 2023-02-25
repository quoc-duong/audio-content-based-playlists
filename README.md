# Audio content based playlists

## Install requirements

```
pip install -r requirements.txt
```

## Analysis of data

This will generate a `descriptors.csv` file in the `analysis` folder.

```
./run_analysis.sh
```

For use in Google Colab, please use the notebook `analysis/audio_content_based_playlist.ipynb`.

## Streamlit application

It will make use of a `descriptors.csv` located in the `analysis` folder.

```
./run_streamlit.sh
```

## AUTHOR
quoc-duong.nguyen
