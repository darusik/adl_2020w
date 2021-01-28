import albumentations as alb
from albumentations.pytorch import ToTensorV2
import re
import torch
import torchvision.models as models
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2RGB
from cnn_finetune import make_model

transform = alb.Compose([
    alb.Resize(64, 64),
    alb.Normalize(),
    ToTensorV2(),
    ])

class MaskDetector():
    def __init__(self, model_path="models/mask_resnet18_epochs_5.pth"):

        search_model_name = re.search(r'.*mask_(.*)_epochs.*', model_path, flags=0)
        assert search_model_name
        model_name = search_model_name.group(1)
        model = make_model(
        model_name,
        pretrained=False,
        num_classes=2,
        dropout_p=0.2,
        input_size=(64, 64) if model_name.startswith(('vgg', 'squeezenet')) else None,
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()
        self.model = model

    def __call__(self, image):
        image = transform(image=image)['image']
        return self.model(image[None, :])

if __name__ == '__main__':

    resnet18_mask_detector = MaskDetector()
    image_path = '/home/darusik/datasets/lfw/lfw_mtcnnpy_160/Aaron_Tippin/Aaron_Tippin_0001.png'
    image = imread(image_path)
    image = cvtColor(image, COLOR_BGR2RGB)
    print(np.argmax(resnet18_mask_detector(image).detach().numpy()))