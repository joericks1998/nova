import os

def save_model(model, dir):
    os.makedirs(dir, exist_ok = True)
    model.save(os.path(dir))
    print("Model Saved Successfully")
