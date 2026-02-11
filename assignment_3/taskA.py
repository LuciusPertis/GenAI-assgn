import gensim
from gensim.models import Word2Vec
import random
import pandas as pd
import os

# --- CONFIGURATION ---
# Set this to True if you have downloaded the Cornell Dataset
USE_REAL_DATASET = True
#DATASET_PATH = r"asssignment_3\dataset_song_cornell\dataset\yes_small" # Path to your downloaded file
DATASET_PATH = r"C:\Users\luciu\dEV\GenAI-assgn\assignment_3\dataset_song_cornell\dataset\yes_small"
#DATASET_PATH = r".\dataset_song_cornell\dataset\yes_small" # Path to your downloaded file

def generate_mock_data():
    """
    Generates fake playlists to demonstrate the concept without needing
    the massive real download immediately.
    """
    print("--- DEMO MODE: Generating 5,000 fake playlists ---")
    
    # A small universe of songs
    genres = {
        "Rock": ["Bohemian Rhapsody", "Stairway to Heaven", "Hotel California", "Imagine", "Smells Like Teen Spirit"],
        "Pop": ["Billie Jean", "Shape of You", "Thriller", "Rolling in the Deep", "Uptown Funk"],
        "Jazz": ["Take Five", "So What", "My Favorite Things", "Fly Me to the Moon", "What a Wonderful World"]
    }
    
    playlists = []
    
    # Create 5000 random playlists
    for _ in range(5000):
        # People tend to stick to one genre in a playlist, but sometimes mix
        main_genre = random.choice(list(genres.keys()))
        playlist_length = random.randint(3, 10)
        
        current_playlist = []
        for _ in range(playlist_length):
            # 80% chance to pick from main genre, 20% random noise
            if random.random() < 0.8:
                song = random.choice(genres[main_genre])
            else:
                random_genre = random.choice(list(genres.keys()))
                song = random.choice(genres[random_genre])
            current_playlist.append(song)
            
        playlists.append(current_playlist)
        
    return playlists

def load_real_dataset(path):
    """
    Load the Yes.com Small dataset (Cornell).
    Expected structure:
    - path/song_hash.txt: id <tab> title <tab> artist
    - path/train.txt: space-separated song IDs
    """
    print(f"Loading dataset from {path}...")
    
    # 1. Load Song Mappings (ID -> "Title - Artist")
    hash_file = os.path.join(path, "song_hash.txt")
    id_to_song = {}
    
    if not os.path.exists(hash_file):
        print(f"Error: {hash_file} not found. Switching to Mock Data.")
        return generate_mock_data()

    print(f"Found {hash_file}. Parsing song_hash.txt...")

    # Variables for T.N.T. check
    tnt_ids = []
    tnt_name = "T.N.T."

    with open(hash_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:

                # Format: id    title   artist
                # eg:"434	Faint	Linkin Park"
                sid = parts[0].strip()
                title = parts[1].strip()
                artist = parts[2].strip() if len(parts) > 2 else "Unknown"
                full_name = f"{title} - {artist}"
                id_to_song[sid] = full_name

                # Check for T.N.T.
                if title == tnt_name:
                    tnt_ids.append(sid)
                    print(f"DEBUG: Found '{tnt_name}' in song_hash.txt with ID: {sid} ({full_name})")

    if not tnt_ids:
        print(f"DEBUG: '{tnt_name}' NOT found in song_hash.txt")

    # 2. Load Playlists (train.txt and test.txt)
    playlists = []
    tnt_found_in_playlists = False

    for fname in ["train.txt", "test.txt"]:
        fpath = os.path.join(path, fname)
        if os.path.exists(fpath):
            print(f"Found {fpath}. Parsing {fname}...")
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    # IDs are space separated in this dataset
                    song_ids = line.strip().split() 
                    
                    # Check if T.N.T. is in this playlist (using ID)
                    for tid in tnt_ids:
                        if tid in song_ids:
                            tnt_found_in_playlists = True

                    # Convert IDs to Song Names
                    song_names = [id_to_song[sid] for sid in song_ids if sid in id_to_song]
                    
                    if len(song_names) > 1:
                        playlists.append(song_names)
        else:
            print(f"Warning: {fpath} not found.")

    if tnt_ids:
        if tnt_found_in_playlists:
            print(f"DEBUG: '{tnt_name}' (IDs: {tnt_ids}) was found in the playlists.")
        else:
            print(f"DEBUG: '{tnt_name}' (IDs: {tnt_ids}) was NOT found in any playlist.")
    
    print(f"Loaded {len(playlists)} playlists.")
    if not playlists:
        print("Warning: No playlists loaded. Returning mock data.")
        return generate_mock_data()
    
    # do a EOD of playlists
    print("\n\n==================================\n")
    for i, playlist in enumerate(playlists):
        print(f"Playlist {i+10}: {playlist}")
        if i >= 4:  # Show only the first 5 playlists for brevity
            break
        
    return playlists

def train_recommender(playlists):
    """
    Trains the Word2Vec model.
    """
    print("Training Word2Vec model (this might take a moment)...")
    
    # SG=1 (Skip-Gram) is usually better for recommendation tasks than CBOW
    # window=5 means we look at 5 songs before and after the current song
    model = Word2Vec(
        sentences=playlists, 
        vector_size=100, 
        window=5, 
        min_count=1, 
        workers=4,
        sg=1 
    )
    print("Training complete.")
    return model

def recommend_songs(model, song_title, top_n=5):
    """
    Finds similar songs in the embedding space.
    """
    print(f"\n--- Recommendations for '{song_title}' ---")
    try:
        recommendations = model.wv.most_similar(song_title, topn=top_n)
        for rank, (song, score) in enumerate(recommendations, 1):
            print(f"{rank}. {song} (Similarity: {score:.4f})")
    except KeyError:
        print(f"Error: '{song_title}' was not found in the training data. Did you mean one of these?")
        # Show song names containing the input as substring
        candidates = [song for song in model.wv.index_to_key if song_title.lower() in song.lower()]
        for candidate in candidates:
            print(f"  - {candidate}")
        

def main():
    # 1. Prepare Data
    if USE_REAL_DATASET:
        # chk path
        if os.path.isdir(DATASET_PATH):
            print(f"Dataset path '{DATASET_PATH}' found. Loading real dataset.")
        else:
            print(f"Dataset path '{DATASET_PATH}' INVALID.")
        playlists = load_real_dataset(DATASET_PATH)
    else:
        playlists = generate_mock_data()

    # 2. Train Model
    # This fulfills: "Train a Song Embedding Model" [cite: 77]
    model = train_recommender(playlists)

    # 3. Test Recommendations
    # We test with a song we know exists in our mock data
    test_song = "T.N.T. - AC/DC"
    recommend_songs(model, test_song)
    
    # Test another genre
    test_song_2 = "Faint"
    recommend_songs(model, test_song_2)

    # 4. Interactive Mode
    while True:
        user_input = input("\nEnter a song name to get recommendations (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            break
        recommend_songs(model, user_input)

if __name__ == "__main__":
    main()

"""
    OUTPUT:

    Enter a song name to get recommendations (or 'q' to quit): Faint

    --- Recommendations for 'Faint' ---
    Error: 'Faint' was not found in the training data. Did you mean one of these?
    - Faint - Linkin Park

    Enter a song name to get recommendations (or 'q' to quit): Faint - Linkin Park

    --- Recommendations for 'Faint - Linkin Park' ---
    1. Remedy - Seether (Similarity: 0.9793)
    2. Sabotage - The Beastie Boys (Similarity: 0.9793)
    3. Crawling - Linkin Park (Similarity: 0.9778)
    4. BYOB - System Of A Down (Similarity: 0.9765)
    5. So Far Away - Staind (Similarity: 0.9763)

    Enter a song name to get recommendations (or 'q' to quit): So Far

    --- Recommendations for 'So Far' ---
    Error: 'So Far' was not found in the training data. Did you mean one of these?
    - So Far Gone - J Boog
    - If Heaven Wasn't So Far Away - Justin Moore
    - So Far Away - Staind

    Enter a song name to get recommendations (or 'q' to quit): So Far Away - Staind

    --- Recommendations for 'So Far Away - Staind' ---
    1. Fine Again - Seether (Similarity: 0.9845)
    2. She Hates Me - Puddle Of Mudd (Similarity: 0.9843)
    3. Remedy - Seether (Similarity: 0.9837)
    4. Little Things - Bush (Similarity: 0.9826)
    5. Through Glass - Stone Sour (Similarity: 0.9805)

"""