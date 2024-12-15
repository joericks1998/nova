import os

def save_model(model = None, save_dir = None):
    os.makedirs(save_dir, exist_ok = True)
    model.save(os.path(save_dir))
    print("Model Saved Successfully")
