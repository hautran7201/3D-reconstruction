import os

def get_folder_names(directory):
    folder_names = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return folder_names