import os
import logging
import threading

tmp_dir = '/tmp'
os.makedirs(tmp_dir, exist_ok=True)

headers = [
    {
        "header": "X-Request-ID",
        "value": "123456"
    }
]

valid_file_mimes = [
    'image/jpeg',
    'image/jpg',
    'image/png'
]

path_model = os.path.join(os.getcwd(), 'models')
print(path_model)

map_models = {
    "1": "image_classifier_vgg16.pth",
    "2": "image_classifier_vgg19.pth",
    "3": "image_classifier_resnet152.pth"
}


log_level = logging.DEBUG


def thread_id_filter(record):
    """Inject thread_id to log records"""
    record.thread_id = threading.get_native_id()
    return record


def logger(name='TEST', level=log_level):
    # create logger
    l = logging.getLogger(name)
    l.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # create formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s]-%(name)s: %(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    l.addHandler(ch)

    return l
