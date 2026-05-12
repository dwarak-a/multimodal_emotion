# Multimodal Emotion Recognition

## 1. Project Overview

The objective of this project is to design a deep learning model capable of classifying human emotions based on the RAVDESS dataset. By utilizing both raw audio files and text transcripts generated via OpenAI's Whisper, this project builds and assesses three models: an audio-only CNN, a text-only DistilBERT model, and a multimodal Early Fusion model.

## 2. Dependencies & Environment

To run this project, the following core packages are required (ignoring standard built-in utilities and sub-dependencies):

**Deep Learning & NLP:**

* `torch` (2.11.0+cu128)

* `torchaudio` (2.11.0+cu128)

* `transformers` (5.8.0)

* `scikit-learn` (1.8.0)

**Audio Processing & Transcription:**

* `openai-whisper` (20250625)

* `soundfile` (0.13.1)

**Data Manipulation & Visualization:**

* `pandas` (3.0.2)

* `matplotlib` (3.10.9)

* `tqdm` (4.67.3)

## 3. Model Architecture

### Text Branch (DistilBERT Upgrade)

The text baseline uses a pre-trained `distilbert-base-uncased` backbone provided by Hugging Face to process the tokenized Whisper transcripts.

* The base DistilBERT transformer weights are entirely frozen to prevent catastrophic forgetting.

* The 768-dimensional `[CLS]` token state outputs to a two-layer Multi-Layer Perceptron (MLP) bottleneck. It maps from `768 -> 128`, and then `128 -> 64` (both utilizing ReLU activations and `0.3` Dropout).

* During unimodal evaluation, this 64-dimensional bottleneck connects to an 8-class Linear output layer.

### Audio Branch (Audio CNN)

The audio branch is built as a standard Convolutional Neural Network processing Mel-spectrogram inputs.

* It contains two sequential block iterations, each mapping: `Conv2D -> BatchNorm2D -> ReLU -> MaxPool2d`.

* An `AdaptiveAvgPool2d` follows, dynamically resizing the spatial map to `(4, 4)`.

* This flattened output passes through the fully connected bottleneck layers (utilizing ReLU and `0.5` Dropout) before entering the final 8-class Linear output layer.

### Multimodal Fusion (Early Fusion)

This system implements the **Early Fusion** pattern.

* The audio and text inputs are independently fed into their respective unimodal backbones.

* Instead of outputting class probabilities, the models output their pre-classification bottleneck features.

* The extracted Audio bottleneck vector and the Text bottleneck vector (size 64) are concatenated along the feature dimension to form a unified, balanced fused vector.

* A final, separate Multi-Layer Perceptron learns the inter-modal correlations from this concatenated state to map the final decision boundary for the 8 emotions.

## 4. Training & Validation Results

*The following metrics evaluate the performance of each model on the withheld test actors (actors 21-24).*

| Model Configuration | Accuracy | Precision | Recall | F1-Score | 
 | ----- | ----- | ----- | ----- | ----- | 
| **Audio-Only CNN** | 39.17% | 0.3772 | 0.3917 | 0.3358 | 
| **Text-Only DistilBERT** | 16.25% | 0.0742 | 0.1625 | 0.0718 | 
| **Multimodal Early Fusion** | 37.50% | 0.3623 | 0.3750 | 0.3329 | 

## 5. Visualizations

*(Note: These charts are generated during training and saved to the `\processed\` folder).*

* **Audio Model Loss:** `![Audio Loss](..\processed\audio_loss_plot.png)`

* **Text Model Loss:** `![Text Loss](..\processed\text_loss_plot.png)`

* **Multimodal Loss:** `![Multimodal Loss](..\processed\multimodal_loss_plot.png)`

## 6. Observations & Conclusion

Based on the quantitative results and the output confusion matrices, several critical behaviors were observed regarding the interplay of audio and text in this specific dataset:

1. **The NLP Dataset Bottleneck:** The Text-Only DistilBERT model suffered a severe mode collapse (16.25% accuracy, 0.07 F1-Score). Looking at its confusion matrix, it predicted class 1 (Calm) for nearly every single audio clip. This happens because in the RAVDESS dataset, actors speak the *exact same two lexical sentences* ("Kids are talking by the door" and "Dogs are sitting by the door") with different emotional tones. Because the transcribed words contain zero emotional variance, the text modality inherently possesses zero predictive power.

2. **Audio Carries the Signal:** The Audio-Only CNN achieved the highest accuracy (39.17%). While predicting 8 granular classes purely from audio is highly complex, the audio spectrograms inherently map vocal inflection, volume, and tempo, making it the only modality in this pipeline actually capable of solving the task.

3. **Multimodal Drag:** In theory, Multimodal models should perform better than unimodal models. However, the Early Fusion model (37.50% Accuracy) actually underperformed the Audio CNN. Because the concatenated text bottleneck features provided constants/noise rather than meaningful variance, they actively diluted the useful audio features, acting as a "drag" on the classifier's performance.

4. **Confusion Matrix Takeaways:** In the Audio CNN, the model was most successful at predicting 'Calm', 'Disgust', and 'Surprised', but heavily confused 'Neutral' (often misclassifying it as 'Calm'). This suggests that low-energy/low-inflection vocal patterns map too closely to one another in the Mel-Spectrogram feature space to be easily distinguishable without heavier temporal sequence modeling.