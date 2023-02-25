import os
import json
import numpy as np
import pandas as pd
import essentia.standard as es
from tqdm import tqdm
import labels  # Discogs genres
import copy
import argparse


def get_mp3_paths(dataset_dir):
    """
    Traverse the dataset directory and retrieve the paths of mp3 files into a list
    """
    os.chdir(dataset_dir)

    # A list to hold the paths of all the mp3 files found
    mp3_paths = []

    # Walk the directory tree and find all mp3 files
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.mp3'):
                rel_path = os.path.relpath(root, dataset_dir)
                mp3_paths.append(os.path.join(rel_path, file))

    return mp3_paths


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        dest="dataset",
        default='../audio/',
        help="Directory containing generated masks folder and copied image folder",
        type=str,
    )

    return parser.parse_args()


def get_tempo(audio, tempo_model):
    global_tempo, local_tempo, local_tempo_probabilities = tempo_model(audio)
    return global_tempo, local_tempo, local_tempo_probabilities


def get_music_styles(audio, styles_model):
    activations = styles_model(audio)
    return activations.mean(axis=0)


def get_voice_instrument(audio, voice_instrument_model):
    """
    Returns:
    (int, int): Instrumental, Voice (presence or absence of voice)
    """
    activations = voice_instrument_model(audio)
    return activations.mean(axis=0)


def get_danceability(audio, danceability_model):
    """
    Returns:
    (int, int): Danceable, Not Danceable
    """
    activations = danceability_model(audio)
    return activations.mean(axis=0)


def get_arousal_valence(audio, embeddings_model, av_model):
    """
    Returns:
    (int, int): valence, arousal
    """
    embeddings = embeddings_model(audio)
    predictions = av_model(embeddings)
    return predictions.mean(axis=0)


def create_models():
    tempo_model = es.TempoCNN(graphFilename="../models/deepsquare-k16-3.pb")

    styles_model = es.TensorflowPredictEffnetDiscogs(
        graphFilename="../models/discogs-effnet-bs64-1.pb")

    voice_instrument_model = es.TensorflowPredictMusiCNN(
        graphFilename="../models/voice_instrumental-musicnn-msd-2.pb")

    danceability_model = es.TensorflowPredictMusiCNN(
        graphFilename="../models/danceability-musicnn-msd-2.pb")

    patch_size = 187
    patch_hop_size = patch_size // 2

    embeddings_model = es.TensorflowPredictMusiCNN(
        graphFilename="../models/msd-musicnn-1.pb",
        input="model/Placeholder",
        output="model/dense/BiasAdd",
        patchSize=patch_size,
        patchHopSize=patch_hop_size
    )

    emo_metadata = json.load(
        open("../models/emomusic-musicnn-msd-2.json", "r"))

    input_layer = emo_metadata["schema"]["inputs"][0]["name"]
    output_layer = emo_metadata["schema"]["outputs"][0]["name"]

    av_model = es.TensorflowPredict2D(
        graphFilename="../models/emomusic-musicnn-msd-2.pb",
        input=input_layer,
        output=output_layer
    )

    return tempo_model, styles_model, voice_instrument_model, danceability_model, embeddings_model, av_model


def process_data(mp3_paths):
    tempo_model, styles_model, voice_instrument_model, danceability_model, embeddings_model, av_model = create_models()
    columns = copy.deepcopy(labels.labels)
    columns.extend(['tempo', 'voice', 'danceability', 'arousal', 'valence'])

    df = pd.DataFrame(columns=columns)

    for path in tqdm(mp3_paths):
        audio = es.MonoLoader(filename=path)()

        # Inference for all descriptors
        global_tempo, _, _ = get_tempo(audio, tempo_model)
        styles_activations = get_music_styles(audio, styles_model)
        instrumental, voice = get_voice_instrument(
            audio, voice_instrument_model)
        danceable, not_danceable = get_danceability(audio, danceability_model)
        valence, arousal = get_arousal_valence(
            audio, embeddings_model, av_model)

        with_voice = (voice + 1 - instrumental) / 2
        danceability = (danceable + 1 - not_danceable) / 2

        df.loc[path] = np.concatenate(
            (styles_activations, [global_tempo, with_voice, danceability, arousal, valence]))

    return df


def main():
    args = parse_args()

    mp3_paths = get_mp3_paths(args.dataset)

    df = process_data(mp3_paths)
    df.to_csv('../analysis/descriptors.csv')


if __name__ == '__main__':
    main()
