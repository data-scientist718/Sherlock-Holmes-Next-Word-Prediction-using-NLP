# Next Word Prediction: Case Study

## Overview

This project aims to develop a robust Next Word Prediction model using the Sherlock Holmes text dataset. The model predicts the most probable word following a given sequence of words, leveraging LSTM neural networks.

## Project Structure

1. **Data Preprocessing**: Text cleaning, tokenization, sequence creation.
2. **Model Building**: Constructing the LSTM-based neural network.
3. **Model Training**: Training the model on the dataset.
4. **Evaluation**: Assessing model performance.
5. **Prediction Function**: Function to predict the next word.
6. **Flask Application**: Web app for interactive predictions.

## Installation

1. Clone the repository.
2. Install the required libraries:
    ```bash
    pip install flask nltk keras tensorflow
    ```
3. Download the `sherlock_holmes.txt` dataset and place it in the project directory.

## Usage

1. **Train the model**: Run the script to preprocess the data, build, and train the model.
    ```bash
    python train_model.py
    ```
2. **Run the Flask application**:
    ```bash
    python app.py
    ```
3. Open a web browser and go to `http://127.0.0.1:5000/`.

## Files

- `Sherlock_Holmes_Next_Word_Prediction.IPYNB`: Script for data preprocessing, model training, and saving.
- `app.py`: Flask application for next word prediction.
- `templates/index.html`: HTML template for the web interface.
- `sherlock-holm.es_stories_plain-text_advs.txt`: Text dataset.

## Summary and Conclusion

The project demonstrates the use of LSTM neural networks for next word prediction. The model shows promising results and can be further improved with more data and fine-tuning.


