import ollama
from transformers import AutoTokenizer

# --- CONFIGURATION ---
# The assignment asks for Phi-3, but we use your local Phi-4 as requested.
#OLLAMA_MODEL = "phi4-mini:3.8b-q4_K_M"
OLLAMA_MODEL = "phi4-mini:latest"  #
# We still download the HF tokenizer to demonstrate understanding of Page 1 [cite: 9]
HF_TOKENIZER_ID = "microsoft/Phi-3-mini-4k-instruct" 

def demonstrate_tokenization(text):
    """
    The assignment emphasizes understanding the Tokenizer[cite: 9].
    Since Ollama hides this, we use the HF library here to show what is happening.
    """
    print(f"\n--- DEMO: Tokenizer Logic (Assignment Page 1) ---")
    print(f"Loading tokenizer for: {HF_TOKENIZER_ID}...")
    
    # Load just the tokenizer (very lightweight, no GPU needed)
    tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_ID)
    
    # Demonstrate splitting input text into tokens 
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    
    print(f"Input Text: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print("-" * 50 + "\n")

def generate_text_ollama(prompt):
    """
    Replicates the 'pipeline' function from the PDF  using Ollama.
    """
    print(f"Generating output for: '{prompt}'...")
    
    # Mapping PDF parameters to Ollama options:
    # max_new_tokens=500  -> num_predict
    # do_sample=False     -> temperature=0 (makes it deterministic)
    
    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={
            "num_predict": 500,  # Limit generation
            "temperature": 0.0   # Select next most probable token (greedy)
        }
    )
    
    # The assignment asks for return_full_text=False. 
    # Ollama returns only the generated part by default, satisfying this.
    return response['response']

def main():
    # 1. Demonstrate Tokenization (Educational Requirement)
    sample_text = "Generative AI is fascinating."
    demonstrate_tokenization(sample_text)

    # 2. Generate three outputs as requested [cite: 44]
    prompts = [
        "Explain the difference between static and dynamic embeddings.", # Relevant to [cite: 9]
        "Write a short python function to calculate the Fibonacci sequence.",
        "What are the main components of a Transformer model?"
    ]

    print(f"--- ASSIGNMENT OUTPUTS (Model: {OLLAMA_MODEL}) ---")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nOutput #{i}:")
        print(f"Prompt: {prompt}")
        output = generate_text_ollama(prompt)
        print(f"Result:\n{output}")
        print("="*60)

if __name__ == "__main__":
    main()

"""OUTPUT:

C:\Users\luciu\dEV\GenAI-assgn>python assignment_2/test_assign2.py

--- DEMO: Tokenizer Logic (Assignment Page 1) ---
Loading tokenizer for: microsoft/Phi-3-mini-4k-instruct...
Input Text: 'Generative AI is fascinating.'
Tokens: ['▁Gener', 'ative', '▁A', 'I', '▁is', '▁fasc', 'in', 'ating', '.']
Token IDs: [3251, 1230, 319, 29902, 338, 21028, 262, 1218, 29889]
--------------------------------------------------

--- ASSIGNMENT OUTPUTS (Model: phi4-mini:latest) ---

Output #1:
Prompt: Explain the difference between static and dynamic embeddings.
Generating output for: 'Explain the difference between static and dynamic embeddings.'...
Result:
Static Embeddings vs Dynamic Embeddings:

1. Static Embeddings: These are pre-trained word vectors that represent words as fixed-length numerical vectors in a high-dimensional space, typically obtained through techniques like Word2Vec or GloVe.

- Pre-Trained Vectors: They use large corpora to learn the vector representations of each unique token (word) independently.
- Fixed Length Representation: Each word is represented by an identical-sized embedding regardless of its context within sentences. For example, "cat" and "cats" would have different embeddings even though they share a common root morpheme ("cat").
- Context-Agnostic: Static Embeddings do not take into account the surrounding words or sentence structure when representing each word.
- Pre-Trained Models: Commonly used models for static embedding generation include Word2Vec, GloVe (Global Vectors), and FastText.

2. Dynamic Embeddings: These are context-dependent representations that change based on their usage within sentences during a particular conversation session in real-time applications like chatbots or virtual assistants.
- Context-Aware Representation: Unlike Static Embeddings which ignore the surrounding words' influence, dynamic embeddings consider both left-to-right (LSTM) and right-to-left dependencies to generate contextualized word vectors. For instance, "bank" would have different representations when used as a financial institution versus river bank based on its context.
- Real-Time Generation: Dynamic embeddings are generated in real-time during conversations or interactions with users by leveraging neural networks like RNNs (Recurrent Neural Networks), LSTMs (Long Short-Term Memory units) and Transformers. These models learn to predict the next word given a sequence of previous words, allowing them to generate contextually relevant representations.
- Continuous Learning: Dynamic embeddings can continuously update their representation based on new interactions or conversations with users.

In summary:

Static Embeddings are pre-trained vectors that represent each unique token independently without considering its surrounding sentence structure. On the other hand, dynamic embeddings provide a flexible and adaptive approach by generating contextualized word vector representations in real-time during user interaction sessions using neural networks like RNNs/LSTMs or Transformers.
============================================================

Output #2:
Prompt: Write a short python function to calculate the Fibonacci sequence.
Generating output for: 'Write a short python function to calculate the Fibonacci sequence.'...
Result:
Certainly! Below is an efficient Python function that calculates and returns the first `n` numbers in the Fibonacci sequence using iteration, which has better time complexity compared to recursion for larger values of `n`.

```python
def fibonacci(n):
    """
    Generate n elements from the Fibonacci series.

    :param n: Number of terms in the Fibonacci sequence required. Must be a positive integer greater than 0.
    :return: List containing first 'n' numbers in the Fibonacci sequence.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input must be an integer greater than zero.")

    fib_sequence = [0, 1]
    for i in range(2, n):
        next_fib = fib_sequence[-1] + fib_sequence[-2]
        fib_sequence.append(next_fib)

    return fib_sequence[:n]

# Example usage:
print(fibonacci(10))
```

This function starts with the first two numbers of the Fibonacci sequence (0 and 1) in a list. It then iterates, starting from index `2` up to but not including `n`, calculating each subsequent number by summing up the last two elements added so far into our series (`fib_sequence`). This approach ensures that we only keep what's necessary for generating further numbers of the sequence and thus maintains optimal space complexity as well.
============================================================

Output #3:
Prompt: What are the main components of a Transformer model?
Generating output for: 'What are the main components of a Transformer model?'...
Result:
The primary elements that constitute an encoder-decoder architecture known as Transformers include:


1. **Multi-Head Self-Attention Mechanism**: This component allows each position in the input sequence to attend to all positions within the same segment, enabling it to capture contextual information from different representation subspaces.


2. **Positional Encoding**: Since there is no inherent notion of order or distance between tokens (like words) for models that process sequences like Transformers need positional encoding added as a vector at each position in input sequence so they can learn relative positions and distances among the elements within an input sequence.



4. **Feed-Forward Neural Networks**: These are fully connected layers applied to every point of attention output, which helps with learning complex functions.  


5. **Normalization Layers (Layer Normalization)**: Applied after self-attention mechanisms in each sub-layer before adding residual connections and followed by a ReLU activation function for non-linearity.



7. **Output Layer**: In the decoder part specifically designed to generate an encoded representation of input sequence, which can then be used as output.       


8. **Loss Function (e.g., Cross-Entropy Loss)**: Used during training time to measure how well predictions match actual labels and guide optimization.

9. **Optimizer (e.g., Adam Optimizer or SGD with Momentum)**: Helps in adjusting the weights of neural networks based on computed gradients from loss function. 
10. **Training Loop**: Involves forward propagation, backward propagation for computing gradient descent updates to minimize losses over epochs using batches.  


These components collectively enable Transformers to process sequences and learn dependencies between elements regardless of their distance within input data effectively.
============================================================

C:\Users\luciu\dEV\GenAI-assgn>
"""