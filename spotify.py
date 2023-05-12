from scipy.spatial.distance import cdist
import difflib
from collections import defaultdict
import os
import numpy as np
import pandas as pd
from yellowbrick.target import FeatureCorrelation
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv("C:\Users\Lenovo\Downloads\data.csv.zip")
genre_data = pd.read_csv("C:\Users\Lenovo\Downloads\data_by_genres.csv")
year_data = pd.read_csv("C:\Users\Lenovo\Downloads\data_by_year.csv")

feature_names = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    "duration_ms",
    "explicit",
    "key",
    "mode",
    "year",
]

X, y = data[feature_names], data["popularity"]

# Create a list of the feature names
features = np.array(feature_names)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

plt.rcParams["figure.figsize"] = (20, 20)
visualizer.fit(X, y)  # Fit the data to the visualizer
visualizer.show()


def get_decade(year):
    period_start = int(year / 10) * 10
    decade = "{}s".format(period_start)
    return decade


data["decade"] = data["year"].apply(get_decade)

sns.set(rc={"figure.figsize": (11, 6)})
sns.countplot(data["decade"])

sound_features = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "valence",
]
fig = px.line(year_data, x="year", y=sound_features)
fig.show()

top10_genres = genre_data.nlargest(10, "popularity")

fig = px.bar(
    top10_genres,
    x="genres",
    y=["valence", "energy", "danceability", "acousticness"],
    barmode="group",
)
fig.show()


cluster_pipeline = Pipeline(
    [("scaler", StandardScaler()), ("kmeans", KMeans(n_clusters=10, n_jobs=-1))]
)
X = genre_data.select_dtypes(np.number)
cluster_pipeline.fit(X)
genre_data["cluster"] = cluster_pipeline.predict(X)


tsne_pipeline = Pipeline(
    [("scaler", StandardScaler()), ("tsne", TSNE(n_components=2, verbose=1))]
)
genre_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=["x", "y"], data=genre_embedding)
projection["genres"] = genre_data["genres"]
projection["cluster"] = genre_data["cluster"]

fig = px.scatter(
    projection, x="x", y="y", color="cluster", hover_data=["x", "y", "genres"]
)
fig.show()

song_cluster_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=20, verbose=False, n_jobs=4)),
    ]
)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data["cluster_label"] = song_cluster_labels


pca_pipeline = Pipeline([("scaler", StandardScaler()), ("PCA", PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=["x", "y"], data=song_embedding)
projection["title"] = data["name"]
projection["cluster"] = data["cluster_label"]

fig = px.scatter(projection, x="x", y="y", color="cluster", hover_data=["title"])
fig.show()


sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=os.environ["SPOTIFY_CLIENT_ID"],
        client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
    )
)


def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q="track: {} year: {}".format(name, year), limit=1)
    if results["tracks"]["items"] == []:
        return None

    results = results["tracks"]["items"][0]
    track_id = results["id"]
    audio_features = sp.audio_features(track_id)[0]

    song_data["name"] = [name]
    song_data["year"] = [year]
    song_data["explicit"] = [int(results["explicit"])]
    song_data["duration_ms"] = [results["duration_ms"]]
    song_data["popularity"] = [results["popularity"]]

    for key, value in audio_features.items():
        song_data[key] = [value]

    return pd.DataFrame(song_data)


def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[
            (spotify_data["name"] == song["name"])
            & (spotify_data["year"] == song["year"])
        ].iloc[0]
        return song_data

    except IndexError:
        return find_song(song["name"], song["year"])


def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print(
                "Warning: {} does not exist in Spotify or in database".format(
                    song["name"]
                )
            )
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)
    song_matrix = np.array(song_vectors)
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    flattened_dict = defaultdict(list)
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict


def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ["name", "year", "artists"]
    song_dict = flatten_dict_list(song_list)
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline["scaler"]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, "cosine")
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs["name"].isin(song_dict["name"])]
    return rec_songs[metadata_cols].to_dict(orient="records")
