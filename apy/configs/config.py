import os

headers = [
    {
        "header": "X-Request-ID",
        "value": "123456"
    }
]

path_model = os.path.join(os.getcwd(), 'models')
print(path_model)

map_models = {
    "1": "vgg16",
    "2": "vgg19",
    "3": "resnet152"
}
