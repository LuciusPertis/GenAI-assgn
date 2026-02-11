import ollama
import numpy as np
from numpy.linalg import norm

# --- CONFIGURATION ---
#MODEL_NAME = "phi4-mini:3.8b-q4_K_M"
MODEL_NAME = "phi4-mini:latest"  # Use the latest version of Phi-4 mini for embeddings

# --- DATA: Robotics Domain Corpus (Min 8 documents) ---
corpus = [
    "Autonomous drones are revolutionizing agriculture by monitoring crop health.",  # Doc 0
    "Industrial robotic arms compile cars with high precision in factories.",        # Doc 1
    "Soft robotics is a field focused on constructing robots from compliant materials.", # Doc 2
    "Human-robot interaction is a key challenge in service robotics.",               # Doc 3
    "Surgical robots allow doctors to perform complex procedures with precision.",    # Doc 4
    "Boston Dynamics creates mobile robots with dynamic balance and agility.",       # Doc 5
    "Reinforcement learning is often used to train robots to walk.",                 # Doc 6
    "Warehouse robots automatically sort and transport packages."                    # Doc 7
]

def get_embedding(text):
    return np.array(ollama.embeddings(model=MODEL_NAME, prompt=text)["embedding"])

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))

def search(query, corpus_embeddings):
    """
    Searches the corpus for the query and returns top results.
    """
    print(f"\nQuerying: '{query}'")
    query_vec = get_embedding(query)
    
    scores = []
    for idx, doc_vec in enumerate(corpus_embeddings):
        score = cosine_similarity(query_vec, doc_vec)
        scores.append((score, idx))
    
    # Sort by score descending (highest similarity first)
    scores.sort(key=lambda x: x[0], reverse=True)
    
    return scores[:3] # Return top 3

def main():
    print(f"--- Q5: Mini Semantic Search Engine ({MODEL_NAME}) ---")
    print("Indexing corpus (generating embeddings)...")
    
    # 1. Indexing: Convert all docs to vectors
    corpus_embeddings = []
    for doc in corpus:
        corpus_embeddings.append(get_embedding(doc))
    print("Indexing complete.\n")

    # 2. Interactive Search
    # Example Query 1
    query1 = "How are robots used in medicine?"
    results1 = search(query1, corpus_embeddings)
    
    print(f"\nTop 3 Results for '{query1}':")
    for score, idx in results1:
        print(f"[Score: {score:.4f}] {corpus[idx]}")

    # Example Query 2
    query2 = "robots in farming"
    results2 = search(query2, corpus_embeddings)
    
    print(f"\nTop 3 Results for '{query2}':")
    for score, idx in results2:
        print(f"[Score: {score:.4f}] {corpus[idx]}")

if __name__ == "__main__":
    main()

"""
    OUTPUT:

    C:\Users\luciu\dEV\GenAI-assgn>python asssignment_3\q5.py
    --- Q5: Mini Semantic Search Engine (phi4-mini:latest) ---
    Indexing corpus (generating embeddings)...
    Indexing complete.


    Querying: 'How are robots used in medicine?'

    Top 3 Results for 'How are robots used in medicine?':
    [Score: 0.8969] Industrial robotic arms compile cars with high precision in factories.
    [Score: 0.8918] Boston Dynamics creates mobile robots with dynamic balance and agility.
    [Score: 0.8765] Reinforcement learning is often used to train robots to walk.

    Querying: 'robots in farming'

    Top 3 Results for 'robots in farming':
    [Score: 0.6221] Boston Dynamics creates mobile robots with dynamic balance and agility.
    [Score: 0.6021] Industrial robotic arms compile cars with high precision in factories.
    [Score: 0.5721] Warehouse robots automatically sort and transport packages.

    C:\Users\luciu\dEV\GenAI-assgn>
"""