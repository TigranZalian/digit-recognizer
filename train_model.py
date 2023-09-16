from tensorflow import keras
from utils import setup_dir
import numpy as np
import pandas as pd
import os
import zipfile


NUM_EPOCHS = 20
MODELS_DIR = 'models'
MODEL_NAME = f'{MODELS_DIR}/digit-recognition-model-cnn'

TRAIN_DATA_PATH = 'data/mnist_train.csv'
TEST_DATA_PATH = 'data/mnist_train.csv'
DATA_ZIP_PATH = 'data.zip'

def process_data(df: pd.DataFrame):
    label_col = 'label'
    y = df[[label_col]].to_numpy(np.int64)
    x = df[[c for c in df.columns if c != label_col]].to_numpy(np.float64) / 255.0
    return x.reshape(x.shape[0], 28, 28), y

def prepare_data():
    if os.path.exists(TRAIN_DATA_PATH) and os.path.exists(TEST_DATA_PATH):
        return

    if not os.path.exists(DATA_ZIP_PATH):
        raise Exception(f'The dataset is corrupted. Make sure {DATA_ZIP_PATH} exists.')
    
    with zipfile.ZipFile(DATA_ZIP_PATH) as archive:
        archive.extractall('data')

def load_data():
    prepare_data()
    train_data = pd.read_csv('data/mnist_train.csv')
    test_data = pd.read_csv('data/mnist_test.csv')
    x_train, y_train = process_data(train_data)
    x_test, y_test = process_data(test_data)
    return x_train, y_train, x_test, y_test

def create_model_cnn() -> keras.Sequential:
    model = keras.Sequential([
        keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(28, 28, 1)
        ),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def create_model_plain() -> keras.Sequential:
    model = keras.Sequential([
        keras.layers.Input((28, 28, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def create_model() -> keras.Sequential:
    if 'cnn' in MODEL_NAME:
        return create_model_cnn()
    return create_model_plain()

def main():
    print('Start...')

    setup_dir(MODELS_DIR)

    print('Laoding the dataset...')

    x_train, y_train, x_test, y_test = load_data()

    print('Finished loading the dataset')

    model = create_model()

    print(f'Started training the model ({MODEL_NAME})...')
    model.fit(x_train, y_train, epochs=NUM_EPOCHS)
    print('Finished the training')

    print('Estimating model accuracy..')
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    print('Saving the model...')
    model.save(filepath=MODEL_NAME)
    print(f'Model was saved successfully to {MODEL_NAME}')

if __name__ == '__main__':
    main()
