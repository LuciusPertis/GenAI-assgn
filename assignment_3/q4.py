import ollama
import numpy as np
from numpy.linalg import norm

#MODEL_NAME = "phi4-mini:3.8b-q4_K_M"
MODEL_NAME = "phi4-mini:latest"  # Use the latest version of Phi-4 mini for embeddings

def get_embedding(text):
    """
    Generates a vector embedding for the given text using local Ollama.
    """
    response = ollama.embeddings(model=MODEL_NAME, prompt=text)
    return np.array(response["embedding"])

def cosine_similarity(vec_a, vec_b):
    """
    Computes Cosine Similarity: (A . B) / (||A|| * ||B||)
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = norm(vec_a)
    norm_b = norm(vec_b)
    return dot_product / (norm_a * norm_b)

def main():
    # 1. The Sentences from Assignment Q4
    sentences = {
        "S1": "Robots can assist humans in daily tasks",
        "S2": "Machines help people in everyday activities",
        "S3": "The stock market crashed yesterday"
    }

    print(f"--- Q4: Generating Embeddings using {MODEL_NAME} ---\n")

    # 2. Compute Embeddings
    embeddings = {}
    for key, text in sentences.items():
        print(f"Embedding {key}...")
        embeddings[key] = get_embedding(text)

    # 3. Compute & Print Similarity Matrix
    print("\n--- Cosine Similarity Matrix ---")
    keys = list(sentences.keys())
    
    # Print Header
    print(f"{'':<5} | {'S1':<10} | {'S2':<10} | {'S3':<10}")
    print("-" * 45)

    for row_key in keys:
        row_str = f"{row_key:<5} | "
        for col_key in keys:
            score = cosine_similarity(embeddings[row_key], embeddings[col_key])
            row_str += f"{score:.4f}     | "
        print(row_str)

    # 4. Interpretation (Required by Assignment)
    print("\n--- Interpretation ---")
    sim_s1_s2 = cosine_similarity(embeddings["S1"], embeddings["S2"])
    sim_s1_s3 = cosine_similarity(embeddings["S1"], embeddings["S3"])
    
    print(f"Similarity S1-S2: {sim_s1_s2:.4f}")
    print(f"Similarity S1-S3: {sim_s1_s3:.4f}")
    
    if sim_s1_s2 > sim_s1_s3:
        print("\nAnalysis: S1 and S2 have a high similarity score because they share semantic meaning ")
        print("(Robots/Machines, assist/help, humans/people), even though the words are different.")
        print("S3 is unrelated (financial topic), resulting in a lower score.")

if __name__ == "__main__":
    main()

"""
    OUTPUT:

    --- Q4: Generating Embeddings using phi4-mini:latest ---

    Embedding S1...
    Embedding S2...
    Embedding S3...

    --- Cosine Similarity Matrix ---
        | S1         | S2         | S3
    ---------------------------------------------
    S1    | 1.0000     | 0.9654     | 0.8803     |
    S2    | 0.9654     | 1.0000     | 0.8744     |
    S3    | 0.8803     | 0.8744     | 1.0000     |

    --- Interpretation ---
    Similarity S1-S2: 0.9654
    Similarity S1-S3: 0.8803

    Analysis: S1 and S2 have a high similarity score because they share semantic meaning
    (Robots/Machines, assist/help, humans/people), even though the words are different.
    S3 is unrelated (financial topic), resulting in a lower score.
    PS C:\Users\luciu\dEV\GenAI-assgn>

"""
