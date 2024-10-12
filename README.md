# Text Generation with LSTM in Python

## Overview

This project implements a text generation algorithm using an LSTM (Long Short-Term Memory) model trained on Shakespeare's works. The model learns the patterns of characters in the text to generate new sequences of characters based on the input it receives. This allows for the creation of text that mimics the style and structure of Shakespearean writing.

The algorithm processes a subset of Shakespeare's text, constructs a character mapping, and trains an LSTM model to predict the next character in a sequence based on previous characters. The trained model can then generate new text based on different sampling temperatures that control the randomness of the predictions.

## Features

- **Character-Based Model**: The model operates at the character level, predicting the next character in a sequence.
- **Dynamic Text Generation**: The generated text can vary based on the temperature parameter, allowing for both deterministic and creative outputs.
- **Easy Customization**: The sequence length and step size for training can be adjusted for different levels of granularity in text generation.
- **Pre-Trained Model**: The project includes functionality for saving and loading the trained model, allowing for easy reuse without retraining.

## Project Structure

The project consists of a single Python script that handles all the processes from loading data to generating text.

```
text_generation/
│
├── text_generator.py   # The main script that contains the model training and text generation logic
└── README.md           # Documentation for the project
```

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- NumPy

### Installing Dependencies

You can install the required dependencies using pip:

```bash
pip install tensorflow numpy
```

## Setup and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/Juliusmidega/text-generation-lstm.git
cd text-generation-lstm
```

### Step 2: Run the Text Generation Script

To train the model and generate text, run the following command:

```bash
python text_generator.py
```

### Step 3: Model Training

The script will download a subset of Shakespeare's text, preprocess it, and train the LSTM model. The training will last for 4 epochs. Once training is complete, the model will be saved as `textgenerator.model.keras`.

### Step 4: Text Generation

After training, the script generates text with varying creativity based on different temperature values. The output is printed directly to the console.

## How the Algorithm Works

1. **Data Loading**: The script downloads Shakespeare's text and processes a specific range of characters for training.
2. **Character Mapping**: A mapping of characters to indices and vice versa is created for training the model.
3. **Model Architecture**: An LSTM model is constructed with 128 units, followed by a dense layer with a softmax activation to predict the next character.
4. **Training**: The model is trained using categorical cross-entropy loss and the RMSprop optimizer.
5. **Text Generation**: The `generate_text` function generates new text sequences by predicting the next character based on the current input sequence. The randomness of the predictions can be controlled using the temperature parameter.

## Example Outputs

The script will print generated text sequences based on different temperature settings:

```
---------0.2---------
<Generated Text at Temperature 0.2>
---------0.4---------
<Generated Text at Temperature 0.4>
---------0.6---------
<Generated Text at Temperature 0.6>
---------0.8---------
<Generated Text at Temperature 0.8>
---------1---------
<Generated Text at Temperature 1.0>
```

## Future Enhancements

- **Hyperparameter Tuning**: Explore different model architectures and training parameters for improved performance.
- **Expanded Dataset**: Incorporate additional texts to diversify the training data.
- **Interactive Text Generation**: Create a web interface for users to interactively generate text with custom prompts.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Author

Developed by [Julius Midega](https://github.com/Juliusmidega3).

