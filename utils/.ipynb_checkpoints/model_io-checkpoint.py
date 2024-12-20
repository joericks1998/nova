import os
from tensorflow.keras.models import load_model
import inspect

model_dir = "/nova.keras"

def save(model = None, save_dir = None):
    os.makedirs(save_dir, exist_ok = True)
    model.save(save_dir+model_dir)
    print("Model Saved Successfully")
    return

def load(save_dir = None):
    return load_model(save_dir+model_dir)

def master_config(func):
    sig = inspect.signature(func)
    return {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
