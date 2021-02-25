import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from IPython.display import clear_output
from scipy.sparse import csr_matrix
from sklearn.neighbors import  NearestNeighbors

#Define Constants
CLIENT_ID = ''
CLIENT_SECRET = ''
username = ''

#Get Access Token
credentials = SpotifyClientCredentials(client_id=CLIENT_ID,client_secret=CLIENT_SECRET)
token = credentials.get_access_token()
sp = spotipy.Spotify(auth=token)


# Define function: given an album uri, get album and track details
def getAlbumTracks(album_id, id_num):
    # Get album details
    results = sp.album(album_id)
    artist = results['artists'][0]['name'].lower()
    album = results['name'].lower()
    genres = sp.artist(results['artists'][0]['uri'])['genres']

    # Get track details
    results = sp.album_tracks(album_id)
    album_tracks = results['items']
    tracks = []

    for track_meta in album_tracks:
        track_feature = sp.audio_features(track_meta['uri'])[0]
        acousticness = track_feature['acousticness']
        danceability = track_feature['danceability']
        energy = track_feature['energy']
        instrumentalness = track_feature['instrumentalness']
        liveness = track_feature['liveness']
        loudness = track_feature['loudness']
        speechiness = track_feature['speechiness']
        tempo = track_feature['tempo']
        valence = track_feature['valence']

        track = [acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence]
        tracks.append(track)

    df = pd.DataFrame(tracks,
                      columns=['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness',
                               'speechiness', 'tempo', 'valence'])
    df = df.mean()

    df2 = pd.DataFrame(
        columns=['id', 'artists', 'album', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness',
                 'loudness', 'speechiness', 'tempo', 'valence', 'genres'],
        data=[[id_num, artist, album, df['acousticness'], df['danceability'], df['energy'], df['instrumentalness'],
               df['liveness'], df['loudness'], df['speechiness'], df['tempo'], df['valence'], genres]])
    return df2

# Define function: given search criteria and database, give recommendations
def getRecommendations(query_index, database):
    cols = list(database)
    cols.remove('artists')
    cols.remove('album')
    cols.remove('genres')

    dataset = database[cols].copy()
    dataset.set_index('id', drop=True, inplace=True)

    # Create sparse matrix with the columns and data
    dataset_matrix = csr_matrix(dataset.values)
    dataset_matrix

    # Create the nearest neighbor model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(dataset_matrix)

    counter = 0

    distances, indices = model_knn.kneighbors(dataset.iloc[query_index, :].values.reshape(1, -1), n_neighbors=15)
    for i in range(0, len(distances.flatten())):
        #  if distances.flatten()[i] <= 0.0001:
        if i == 0:
            print('Recommendations for {0}:\n'.format(
                database[database['id'] == (dataset.index[query_index])]['album'].values[0]))
        else:
            if database[database['id'] == (dataset.index[indices.flatten()[i]])]['artists'].values[0] != artist_in:
                search_album = database[database['id'] == (dataset.index[indices.flatten()[i]])]['album'].values[0]
                search_artist = database[database['id'] == (dataset.index[indices.flatten()[i]])]['artists'].values[0]
                print(search_album + ' by ' + search_artist)
                search_string = 'artist:' + search_artist + ' album:' + search_album
                #print(sp.search(search_string, type="album")['albums']['items'][0]['images'][0]['url'])

                counter += 1
                if counter == 3:
                    break

# Official code starts here
database = pd.read_csv('albums_with_genres.csv',index_col=None)

# Prompt user to provide artist and album
artist_in = input("Enter Artist: ")
album_in = input("Enter Album title: ")
artist_in = artist_in.lower()
album_in = album_in.lower()

if database['artists'].isin([artist_in]).any() and database['album'].isin([album_in]).any():
    query_index = database[(database['artists'] == artist_in) & (database['album'] == album_in)].index.values[0]
    print('called recommendation function')
    getRecommendations(query_index, database)

else:
    print('Not found in databse, adding...')
    # Grab album details from Spotify
    search_string = 'artist:' + artist_in + ' album:' + album_in
    results = sp.search(q=search_string, type="album")
    id_num = len(database.index)
    new_album = getAlbumTracks(results['albums']['items'][0]['uri'], id_num)

    genre_list = new_album['genres'].tolist()
    genre_list = genre_list[0]

    # Clean result, add genre dummies, and add to database
    cols = list(database)
    # cleaned_album = getGenreDummies(new_album,cols)
    cols.remove('id')
    cols.remove('artists')
    cols.remove('album')
    cols.remove('acousticness')
    cols.remove('danceability')
    cols.remove('energy')
    cols.remove('instrumentalness')
    cols.remove('liveness')
    cols.remove('loudness')
    cols.remove('speechiness')
    cols.remove('tempo')
    cols.remove('valence')
    cols.remove('genres')

    zeroes = np.zeros(shape=(1, len(cols)))
    genre_df = pd.DataFrame(zeroes, columns=cols)
    merged_df = new_album.join(genre_df)
    database = pd.concat([database, merged_df], ignore_index=True)

    for genre in genre_list:
        if genre in cols:
            database[genre].iloc[id_num] = 1
        else:
            database[genre] = 0
            database[genre].iloc[id_num] = 1

    query_index = database[(database['artists'] == artist_in) & (database['album'] == album_in)].index.values[0]
    getRecommendations(query_index, database)