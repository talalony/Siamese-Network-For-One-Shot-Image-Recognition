# Siamese Neural Network for One-Shot Image Recognition

This project is an implementation of the paper ["Siamese Neural Networks for One-Shot Image Recognition"](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) by Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov. The goal is to create a Siamese network that can learn to differentiate between different classes of images with minimal training examples, making it ideal for one-shot learning scenarios.

## Overview

One-shot learning aims to recognize an object category based on a single example from each possible class. This project uses a Siamese Neural Network, a type of neural network that learns to differentiate between pairs of inputs, to perform one-shot image recognition. The model is trained on the Omniglot dataset, which contains images of handwritten characters from different alphabets.

## Dataset

The Omniglot dataset, consisting of two parts, **images_background** and **images_evaluation**, is used in this project. You can download the dataset from [Kaggle: Omniglot Dataset](https://www.kaggle.com/datasets/qweenink/omniglot).

## Changes I Made

- Instead of using layer-wise learning rate and momentum with SGD optimizer I used Adam optimizer with a single learning rate for the entire network.
- Instead of using 12 out of 20 drawers for each alphabet in the training set, I used all 20.

## Installation

To run the project, you need Python and the following packages:

- matplotlib
- Pillow
- torch
- torchvision
- tqdm

## Usage

- Clone the repository and install the necessary dependencies:
```
git clone https://github.com/talalony/Siamese-Network-For-One-Shot-Image-Recognition.git
cd Siamese-Network-For-One-Shot-Image-Recognition
pip install -r requirements.txt
```

- To train the model, run:
```
python main.py --train "path/to/images_background"
```

- To test the model, use:
```
python main.py --test "path/to/images_evaluation"
```

## Results

Loss value is recorded after each epoch

![Figure_1](https://github.com/user-attachments/assets/be3f5bda-da93-45e3-93fe-9d15b363971f)

The model achieved an accuracy of 92% on the test set. This accuracy is slightly lower than that reported in the original article, likely due to the changes I mentioned.
