from setuptools import setup, find_packages

setup(
    name="assignment2_local",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "ollama",       # To communicate with your local model
        "transformers", # To demonstrate the Tokenizer part of the assignment [cite: 9]
        "torch"         # Required by transformers
    ],
)

