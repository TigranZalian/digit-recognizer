import os
import time
import uuid
import numpy as np


def utc_now() -> int:
    return int(time.time() * 1000)

def unique_id() -> str:
    return str(uuid.uuid4())

def minmax(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def setup_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
