# Digit Recognizer

## Overview

This is a Python project that serves as my "Hello World" introduction to the field of machine learning. In this project, I will build a digit recognizer using the famous MNIST dataset. The goal is to train a machine learning model (CNN - convolutional neural network) that can accurately classify handwritten digits from 0 to 9. Additionally, users can test the trained model in real-time using the playground.

## Project Structure

The project is organized as follows:

```
digit-recognizer/
│
├── html/
│   └── index.html 
│
├── models/
|   └── digit-recognition-model-cnn
|
├── data.zip
├── api.py
├── train_model.py
├── utils.py
│
├── serve.sh
│
├── README.md
└── requirements.txt
```

- **data.zip**: The archive that contains the training and testing datasets in CSV format.
- **html**: Contains the main view of the digit recognition playground.
- **models**: This directory stores the trained machine learning models.
- **api.py**: This file defines the API interface.
- **train_model.py**: Utilize this file for model training.
- **utils.py**: Contains various utility functions.
- **README.md**: You are currently reading this documentation.
- **requirements.txt**: Lists the project's dependencies.
- **images**: This folder will appear once you interact with the trained model using the playground. It contains the images of digits you have drawn for the model's real-time testing.

## Getting Started

1. Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/digit-recognizer.git
```

2. Navigate to the project directory:

```bash
cd digit-recognizer
```

3. Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Uncompress [data.zip](data.zip)

2. Train the model:

```bash
python train_model.py
```

3. Launch the digit recognition playground to interact with the trained model:

```bash
sh serve.sh
```

4. Open http://127.0.0.1:3000 to access the playground and test the model in real-time.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project draws inspiration from the MNIST dataset, a widely recognized dataset for digit recognition in the machine learning community.

Enjoy your journey into the world of machine learning and digit recognition! If you have any questions or need assistance, feel free to reach out. Happy coding!
