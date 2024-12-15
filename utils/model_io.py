import os

def save_model(model = None, save_dir = None):
    os.makedirs(save_dir, exist_ok = True)
    model.save(save_dir+"/nova.keras")
    print("Model Saved Successfully")
    return
