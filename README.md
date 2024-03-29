# Voice Emotion Detection

This project implements a Voice Emotion Detection system using machine learning techniques, specifically leveraging a Long Short-Term Memory (LSTM) neural network model. The system comprises a graphical user interface (GUI) for real-time emotion recognition and a machine learning model trained on the Toronto Emotional Speech Set (TESS) dataset.

## Files

- `gui.py`: Python script containing the GUI for real-time emotion detection.
- `model_creation.ipynb`: Jupyter notebook for creating and training the emotion detection model.
  
## Instructions

1. Run the `gui.py` script to launch the GUI for real-time emotion detection.
2. Ensure the model files (`best_model.json` and `best_model1.h5`) are present in the root directory.

## Real-time Emotion Detection

- Click the "Upload Voice File" button to select an audio file for emotion detection.
- Click the "Start Recording" button to record your voice and get real-time emotion predictions.

**Output Sample:**

![Screenshot (104)](https://github.com/monalisaburma/voice_emotion_detection/assets/122416015/a21ed31c-35a8-451a-8355-c45b386f8968)

## Model

The emotion detection model is trained on the Toronto Emotional Speech Set (TESS) dataset. For details on the model architecture and training process, refer to `model_creation.ipynb`.

## Dataset

The training dataset used for this project is the Toronto Emotional Speech Set (TESS). You can find the TESS dataset [here](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess).

## NullClass Internship

This project is part of the NullClass Internship.

Feel free to explore, contribute, and provide feedback!

