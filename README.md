# Music Genre Classification

This project focuses on building two different pipelines for classifying music genres from audio clips using machine learning and deep learning techniques. We use both handcrafted features with traditional neural networks and raw audio input with a transformer model.

## Objective

The goal is to classify 30-second music clips into one of 10 predefined genres using models that can either learn from statistical audio features or directly from raw waveforms. We compare both approaches based on performance metrics like accuracy and F1 score.

## Dataset

- Dataset used: [GTZAN Music Genre Dataset (Kaggle)](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- Contains 1000 audio clips, each 30 seconds long
- 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

## Methodology

### Pipeline A: Traditional Deep Learning with Keras

1. Extracted audio features like MFCCs, tempo, and chroma statistics.
2. Standardized the features using StandardScaler.
3. Reduced dimensionality using PCA for visualization.
4. Trained multiple dense neural network models.
5. Best performing model was a 5-layer MLP with dropout regularization.

### Pipeline B: Transformer-Based Model using Wav2Vec2

1. Raw audio files were resampled to 16kHz and silence was removed.
2. Used Facebook's [Wav2Vec2-base-960h model](https://huggingface.co/facebook/wav2vec2-base-960h) with Hugging Face’s Trainer API.
3. Balanced the dataset using RandomOverSampler.
4. Trained the model on raw waveforms directly.

## Evaluation

### Keras-Based Models

- Test Accuracy: 93.42%
- Validation Accuracy: 94.03%
- Macro F1 Score: 93.47%

### Transformer-Based Model (Wav2Vec2)

- Test Accuracy: 64.5%
- Macro ROC AUC Score: 0.93

## Comparative Analysis

- Keras-based models achieved higher accuracy and performed well with statistical features.
- Transformer-based model showed strong probabilistic outputs (ROC AUC), which is useful for real-world confidence-based systems.
- This comparison helps understand the trade-off between handcrafted features and end-to-end learning from raw audio.

## Conclusion

We successfully implemented two different approaches for music genre classification. The traditional model using extracted features achieved better classification accuracy, while the transformer-based model was effective in learning directly from raw audio. This project highlights the importance of feature engineering and also shows the potential of modern deep learning models in audio classification tasks.

thank you
