import cv2


def draw_box_on_image(img, boxes, color=(255, 0, 0)):
    """
    Plots predicted bounding pred_boxes, gt_boxes on the image
    
    Params
    image: cv2 image. [h, w, c]
    boxes: Bounding boxes of an image. [[c, pc, x, y, w, h], ...]
    color: Box color. [r, g, b]
    """

    class_list = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", 
        "bus", "car", "cat", "chair", "cow", 
        "diningtable", "dog", "horse", "motorbike", "person", 
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    for bbox in boxes:
        img_h, img_w = img.shape[0:2]
        x_center, y_center, w, h = bbox[2:]
        x_min, x_max, y_min, y_max = int((x_center - w/2.0)*img_w), int((x_center + w/2.0)*img_w), int((y_center - h/2.0)*img_h), int((y_center + h/2.0)*img_h)
        class_name = str(int(bbox[0]))
        confidence = str(round(bbox[1], 3))
    
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
        
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)    
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + 130, y_min), color, -1)
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            thickness=2,
            color=(255, 255, 255), 
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text=confidence,
            org=(x_min + 40, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7, 
            thickness=2,
            color=(255, 255, 255), 
            lineType=cv2.LINE_AA,
        )