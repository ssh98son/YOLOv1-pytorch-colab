import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np




class ResizePreprocess(object):
    '''
    Preprocess used for validation and test.
    1. resize
    2. ImageNet Z-Score Normalization
    3. ToTensor: numpy (H,W,C) -> pytorch (C,H,W) 
    '''
    def __init__(self, img_size=(448, 448)):
        self.transforms = A.Compose([
            A.Resize(*img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    
    def __call__(self, img, bboxes):
        # Albumentations transforms
        class_labels = [bbox[0] for bbox in bboxes]
        bboxes_labels = [bbox[1:5] for bbox in bboxes]
        transformed = self.transforms(image=img, bboxes=bboxes_labels, class_labels=class_labels)
        img = transformed['image']
        
        # Collect converted bboxes
        bboxes = []
        num_bboxes = len(transformed['bboxes'])
        for i in range(num_bboxes):
            bbox = [transformed['class_labels'][i]] + list(transformed['bboxes'][i])
            bboxes.append(bbox)

        return img, bboxes


class JitterPreprocess(object):
    '''
    Preprocess used for online training set augmentation.
    1. resize
    2. color jittering
    3. flip
    4. crop
    5. gauss noise
    6. rotation & translation
    7. ImageNet Z-Score Normalization
    8. ToTensor: numpy (H,W,C) -> pytorch (C,H,W) 
    '''
    def __init__(self, img_size=(448, 448)):
        # Use ImageNet mean value for padding
        self.pixel_mean = [int(v * 255.0) for v in [0.485, 0.456, 0.406]]

        # Albumentations pipeline
        self.transforms = A.Compose([
            A.Resize(*img_size),
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.6, hue=0.15, p=1),
            A.HorizontalFlip(p=0.5),
            A.RandomSizedBBoxSafeCrop(*img_size, erosion_rate=0.3, p=1.0),
            A.GaussNoise(p=0.25),
            A.ShiftScaleRotate(shift_limit=[-0.2, 0.2], scale_limit=[-0.5, 0.2], rotate_limit=[-15, 15], border_mode=cv2.BORDER_CONSTANT, value=self.pixel_mean, p=0.95),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.2, label_fields=['class_labels']))


    def __call__(self, img, bboxes):
        '''
        Takes OpenCV RGB image and returns jittered and normalized image tensor.
        
        Params
        - img: OpenCV RGB Image
        - bboxes: list type bboxes

        Rturn
        - img: Jittered/normalized image
        - bboxes: J
        '''
        # Albumentations transforms
        class_labels = [bbox[0] for bbox in bboxes]
        bboxes_labels = [bbox[1:5] for bbox in bboxes]
        transformed = self.transforms(image=img, bboxes=bboxes_labels, class_labels=class_labels)
        img = transformed['image']

        # Collect converted bboxes
        bboxes = []
        num_bboxes = len(transformed['bboxes'])
        for i in range(num_bboxes):
            bbox = [transformed['class_labels'][i]] + list(transformed['bboxes'][i])
            bboxes.append(bbox)

        return img, bboxes
        

def normalize_imagenet_to_cv2(norm_image):
    '''
    This function creates copy and renormalize to original.
    '''
    
    rgb_image = norm_image.clone().permute(1, 2, 0).numpy()

    pixel_mean = [int(v * 255.0) for v in [0.485, 0.456, 0.406]]
    pixel_std = [int(v * 255.0) for v in [0.229, 0.224, 0.225]]
    
    for c in range(3):
        rgb_image[..., c] = rgb_image[..., c] * pixel_std[c] + pixel_mean[c]
        
    
    rgb_image = np.clip(np.rint(rgb_image).astype(np.uint8), 0, 255)
    rgb_image = np.ascontiguousarray(rgb_image, dtype=np.uint8)
    return rgb_image



########
# Test jittering
########
import cv2
from matplotlib import pyplot as plt

from .visualize import draw_box_on_image


if __name__ == "__main__":
    # load sample image
    img_path = "./toyset/images/000009.jpg"
    label_path = "./toyset/labels/000009.txt"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes = []
    with open(label_path) as f:  
        for label in f.readlines():
            label = label.replace("\n", "").split()
            class_label = int(label[0])
            x = float(label[1])
            y = float(label[2])
            width = float(label[3])
            height = float(label[4])
            boxes.append([class_label, x, y, width, height])

    # jitter
    transform = JitterPreprocess()

    for i in range(10):
        t_img, t_label = transform(img, boxes)
        rgb_img = normalize_imagenet_to_cv2(t_img)
        rgb_label = [obj[0:1] + [1.] + obj[1:] for obj in t_label] # put confidence
        draw_box_on_image(rgb_img, rgb_label)

        plt.figure()
        plt.axis('off')
        plt.imshow(rgb_img)
        plt.show()
