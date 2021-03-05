import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split


def knn_classifier():
    spotify = pd.read_csv('spotify-classlabels-kmeans.csv', delimiter=",")
    target_labels = spotify['cluster']
    spotify = spotify.drop(columns=['cluster'])

    train, test, target_train, target_test = train_test_split(spotify, target_labels, test_size=0.2, random_state=33)

    pca_names = np.array(
        ['mode_0',
         'acousticness',
         'explicit_1',
         'instrumentalness',
         'key',
         'valence'])
    train_numeric = train[pca_names]

    n_neighbors = 37
    knn_classifier = neighbors.KNeighborsClassifier(n_neighbors)
    knn_classifier.fit(train_numeric, target_train)

    return knn_classifier


create_new_playlist = "y"

while create_new_playlist == "y":
    print("""
    ********************************************************************************************************
    Welcome to the Playlist generator using song data from Spotify. First, choose your method of clustering:
    ******************************************************************************************************** 
    """)
    print("""
                What cluster model would you like to use? 
                (1) kmeans with PCA
                (2) kmeans without PCA
                (3) DBSCAN with PCA
                (4) DBSCAN without PCA
                """)
    cluster_method = input("Enter cluster method: ")

    if cluster_method == '1':
        spotify = pd.read_csv('spotify-classlabels-kmeans.csv', delimiter=",")
    elif cluster_method == '2':
        spotify = pd.read_csv('spotify-classlabels-kmeans-without_pca.csv', delimiter=",")
    elif cluster_method == '3':
        spotify = pd.read_csv('spotify-classlabels-DBSCAN.csv', delimiter=",")
    elif cluster_method == '4':
        spotify = pd.read_csv('spotify-classlabels-DBSCAN-without_pca.csv', delimiter=",")

    sample = spotify.sample(10)
    sample = sample[['artists', 'name', 'year', 'cluster']]
    print("Do you want to generate playlist based on an existing song, or enter your own?")
    own_or_existing = input("(n) for new song, (e) for existing: ")

    if own_or_existing == 'e':

        print("""
        Choose the song by entering corresponding number from the following list which will be used 
        to generate playlist of similar songs, (l) to see new list, (q) to quit.
        (Note- using your own song will only use kmeans with PCA clusters for the classification)
        """)
        user_input = None

        while user_input != 'q':
            for i in range(0, 10):
                print(f"""
                ({i})
                Artist: {sample.iloc[i].artists}
                Song: {sample.iloc[i]['name']}
                Decade: {sample.iloc[i].year}
                """)

            user_input = input("Enter song number (or n to input your own song): ")

            if user_input == 'l':
                sample = spotify.sample(10)
                sample = sample[['artists', 'name', 'year']]
                continue

            elif user_input in '0123456789':
                user_input = int(user_input)
                pd.set_option('display.expand_frame_repr', False)

                playlist = spotify.loc[spotify['cluster'] == sample.iloc[user_input].cluster].sample(20)
                playlist = playlist.reset_index()
                print("""
                PLAYLIST
                -------------------------------------------------------------------------------------------------------""")
                print(playlist[['artists', 'name', 'year']])
                create_new_playlist = input("Create new playlist? (y/n)")
                break

    elif own_or_existing == 'n':
        spotify = pd.read_csv('spotify-classlabels-kmeans.csv', delimiter=",")
        name = input("Song name: ")
        artist = input("Artist: ")
        mode = input("Major (1) or minor (0): ")
        acousticness = input("Level of acousticness (from 0-1): ")
        explicit = input("Explicit content (1) or not (0): ")
        instrumentalness = input("Level of instrumentalness (from 0-1): ")
        key = input("Key (approximate from 0-1 where C is 0 and B is 1, i.e C#=.09): ")
        valence = input("Valence (musical happiness level): ")
        # here is where we put the knn classifier to work
        df = pd.DataFrame({
            "mode_0": [int(mode)],
            "acousticness": [float(acousticness)],
            "explicit_1": [int(explicit)],
            "instrumentalness": [float(instrumentalness)],
            "key": [float(key)],
            "valence": [float(valence)]
        })
        knn_clf = knn_classifier()
        prediction = knn_clf.predict(df)
        print(prediction)
        playlist = spotify.loc[spotify['cluster'] == prediction[0]].sample(20)
        playlist = playlist.reset_index()
        pd.set_option('display.expand_frame_repr', False)
        print(playlist[['artists', 'name', 'year']])

        create_new_playlist = input("Create new playlist? (y/n)")


print("Exiting program...")