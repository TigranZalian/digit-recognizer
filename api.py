from fastapi import FastAPI, UploadFile, Response
from fastapi.exceptions import RequestValidationError
from PIL import Image, ImageEnhance, UnidentifiedImageError
from tensorflow import keras
from utils import unique_id, minmax, setup_dir
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import base64


MODEL_NAME = 'models/digit-recognition-model-cnn'
IMAGES_DIR = 'images'
ENABLE_PLOT = True
SAVE_IMAGES = True

app = FastAPI()
model: keras.Sequential = keras.models.load_model(MODEL_NAME)

plt.switch_backend('Agg')
setup_dir(IMAGES_DIR)

with open('html/index.html', 'r') as f:
    index_view = f.read()

def get_plot_data_url(probabilities, predicted_label):
    indices = np.arange(len(probabilities))

    plt.axes().set_xticks(indices)
    bar = plt.bar(indices, probabilities, color="#777777")
    bar[predicted_label].set_color('#566E3D')
    plt.grid(False)
    plt.ylim([0, 1])

    plt.xlabel('Label')
    plt.ylabel('Probability')
    plt.title('Probability Distribution')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return f'data:image/png;base64,{plot_url}'

def process_image(image_stream: BytesIO, id_: str) -> np.ndarray[np.floating]:    
    image = Image.open(image_stream)
    bw_image = image.convert('RGB').convert('L')
    bw_image_enhanced = ImageEnhance.Contrast(bw_image).enhance(10)
    final_image = bw_image_enhanced.resize((28,28))

    if SAVE_IMAGES:
        final_image.save(f'{IMAGES_DIR}/digit_{id_}.png')

    image_array = np.array(final_image, dtype=np.float64).reshape(28, 28, 1)
    image_array = 255.0 - image_array
    image_array = (image_array / image_array.max()) * 255.0
    image_array = image_array / 255.0

    return image_array

def make_digit_predictions(image_array: np.ndarray[np.floating]):
    x = np.array([image_array])
    results = model.predict(x)
    normalised_results = minmax(results)
    return normalised_results[0]

@app.get('/')
def index_endpoint():
    return Response(content=index_view, headers={'Content-Type': 'text/html'})

@app.post('/detect-digit')
def detect_digit_endpoint(image: UploadFile):
    req_id = unique_id()
    image_stream = BytesIO(image.file.read())

    try:
        image_array = process_image(image_stream, req_id)
    except UnidentifiedImageError:
        raise RequestValidationError('Invalid image')

    results = make_digit_predictions(image_array)
    digit = int(np.argmax(results))
    plot_data_url = get_plot_data_url(results, digit) if ENABLE_PLOT else None

    return {
        'id': req_id,
        'detected': digit,
        'results': results.tolist(),
        'plot_data_url': plot_data_url
    }
