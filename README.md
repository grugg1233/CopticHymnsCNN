# Coptic Hymns Audio Classifier (Work in Progress)

This project explores whether a convolutional neural network (CNN) can learn meaningful representations of long-form liturgical audio—specifically Coptic Orthodox hymns—and classify them into their respective hymn categories. The long-term goal is to use this system as the foundation for a broader research program on representation geometry, neural collapse, and subspace structure in audio models.

The current version implements:

- A preprocessing pipeline for long MP3 recordings  
- Sliding-window mel-spectrogram extraction  
- A lightweight CNN trained from scratch  
- A training/validation split with checkpointing  
- A test harness for evaluating on unseen recordings  

This codebase acts as the sandbox where experiments on feature geometry and layer-wise collapse will be conducted.

---

## Motivation

Liturgical music, especially Coptic hymnology, exhibits stable rhythmic, tonal, and structural properties. These patterns make it a natural testbed for studying hierarchical representation learning.

This project is built around several research questions:

- Do early-layer CNN features learned on hymnological audio exhibit Neural Collapse behavior?
- Does collapse emerge differently for religious vs. secular audio?
- Can transfer learning reveal whether certain musical traditions share deeper geometric subspaces in their representations?
- Does fine-tuning on small, hand-labeled datasets preserve or destroy collapse-type structure?

By instrumenting the model with layer hooks and analyzing NC-1/NC-2 metrics on embeddings, we can connect empirical structure to theoretical predictions.

---

## Project Overview

### 1. Dataset

The initial dataset consists of long-form recordings of four Coptic hymns:

- Golgotha  
- Je Nai Nan  
- Tai Shori  
- Ti Shori  

Each recording is processed into overlapping 5-second mel-spectrogram snippets, which form the training samples. A separate test set contains additional recordings not seen during training.

Future expansions include:

- Collecting secular and religious music across multiple traditions  
- Automatically labeling genre/style using web-scraping + heuristics  
- Creating a larger benchmark corpus suitable for transfer learning  

---

### 2. Model

A simple CNN is used as the backbone:

- 1-channel mel-spectrogram input  
- 4 convolutional blocks with batch normalization  
- Global average pooling  
- Fully connected classification head  

This architecture is intentionally minimal so that:

- early-layer features are easy to extract  
- collapse metrics (NC-1 through NC-4) can be computed cleanly  
- transfer learning experiments remain interpretable  

---

### 3. Training

The training loop supports:

- mel-spectrogram extraction with normalization  
- shuffled minibatches  
- validation after each epoch  
- checkpointing every N epochs  
- Reduce-on-Plateau learning rate scheduling  

All checkpointed models are saved into:

