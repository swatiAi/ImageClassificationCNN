
# Convolutional Neural Network (CNN) Image Classifier

Welcome to the CNN Image Classifier project! This project demonstrates the implementation of a Convolutional Neural Network for image classification, with support for CIFAR-10 dataset and a graphical user interface (GUI) for ease of use.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [GUI Usage](#gui-usage)
- [Contributing](#contributing)

## Overview

This project is designed to classify images using a Convolutional Neural Network (CNN). The models included have been trained on the CIFAR-10 dataset and can be used to classify images into one of ten categories.

## Features

- **Pre-trained Models**: Includes pre-trained models ready for use.
- **Customizable Training**: Train your own models using the provided scripts.
- **Graphical User Interface (GUI)**: Easy-to-use interface for image classification.
- **Multiple Models**: Choose from different models (`cifar10_model.h5`, `ImageClassifier.h5`, `best_model.h5`).

## Installation

1. **Clone the Repository**

    \`\`\`bash
    git clone https://github.com/swatiAi/UsedCarEvaluation.git
    cd UsedCarEvaluation
    \`\`\`

2. **Install Dependencies**

    Ensure you have Python and the required libraries installed. You can install the dependencies using:

    \`\`\`bash
    pip install -r requirements.txt
    \`\`\`

## Usage

### Running the CNN Script

To run the CNN script and classify images, use:

\`\`\`bash
python CNN.py
\`\`\`

### Using the GUI

To launch the GUI for image classification, use:

\`\`\`bash
python CNN_GUI.py
\`\`\`

## Model Details

- **cifar10_model.h5**: Pre-trained model on CIFAR-10 dataset.
- **ImageClassifier.h5**: Another version of the model trained on CIFAR-10.
- **best_model.h5**: The best performing model from our training experiments.

## GUI Usage

The GUI allows you to easily classify images without writing any code. Simply launch the GUI and follow the instructions to load and classify images.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.
