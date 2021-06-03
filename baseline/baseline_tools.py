from datetime import datetime
import os
import pickle


def logging(info: str):
    print('\n\r' + '[INFO]' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
          '\n\r' + str(info))


def get_time() -> str:
    return str(datetime.now().strftime("%m-%d-%H-%M"))


def load_pkl_obj(path):
    logging(f'loading pkl obj from {path}')
    if os.path.exists(path):
        with open(path, 'rb') as file:
            return pickle.load(file)
    return None

def save_pkl_obj(obj, path):
    logging(f'saving pkl obj to {path}')
    with open(path, 'wb') as file:
        pickle.dump(obj, file)