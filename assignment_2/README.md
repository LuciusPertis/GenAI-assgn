# Assignment 2 - Local Ollama Setup

## Overview
I used a local **Ollama** server instead of loading the full model weights into Python memory. This is efficient and works with my existing `phi4-mini` model.

Assignment asks for `Microsoft/Phi-3-mini-4k-instruct`but i am using `phi4-mini`. While functionally similar, the output will be from a newer model.

## Prerequisites
1.  **Ollama Running:** Ensure Ollama is running in the background.
    * Verify by running `ollama list` in your terminal.
    * To download just run `ollama phi4-mini:3.8b-q4_K_M`
   
2.  **Python Installed:** Ensure you have Python 3.8+ installed.

## Installation
1.  Open your terminal/command prompt in this folder.
2.  Install the required libraries using the setup script:
    ```bash
    pip install .
    ```
<!-- 
## Usage
Run the assignment script:
```bash
python test_assign2.py -->

## Assignment requirements
1. `max_new_tokens=500`  -> Ollama: `num_predict: 500`
2. `do_sample=False`  -> Ollama: `temperature: 0` (Deterministic)
3. `return_full_text=False`  -> Logic: We print only the response, not the input prompt