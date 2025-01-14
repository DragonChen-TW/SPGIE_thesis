## Introduction

This repository contains the code for my master’s thesis titled "Local Image Explainer Based on Superpixel Algorithms and Graph Neural Networks."

In this work, we use superpixel algorithms to transform digital images into graph representations, which are then used to train a Graph Neural Network (GNN). By integrating our proposed explainer module, the GNN can provide interpretable evidence supporting its decisions.

Finally, we convert the graph-based representations back into pixel space to generate visual explanations. These explanations retain the structural information from the superpixel algorithm, offering more precise details and higher-quality insights.

## Project Structure

This project contains two main parts: `training` and `explaining`.

The checkpoint files are stored under `checkpoints` folder.
The generated dataset files are stored under `dataset` folder.

Both of `checkpoints` and `dataset` are not tracked by Git. 
You can download them through my cloud drive(WIP).

### Training

pass

### Explaining

You can directly checkout the .ipynb files under `explainer` folder.

- [SPGIE explainer on MNIST](explainer/explain_mnist.ipynb)
- [SPGIE explainer on MNIST-M](explainer/explain_mnist.ipynb)
