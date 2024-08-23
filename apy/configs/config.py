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
    "1": "image_classifier_vgg16.pth",
    "2": "image_classifier_vgg19.pth",
    "3": "image_classifier_resnet152.pth"
}
