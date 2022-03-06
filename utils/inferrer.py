import torch


from .metric import Metric
from .box_processes import BoxProcesses


class Inferrer:

    def __init__(self, DEVICE="cuda"):
        self.box_processor = BoxProcesses()
        self.DEVICE = DEVICE


    def infer_labeled(self, loader, model, iou_threshold, threshold,
                pred_format="cells", box_format="midpoint"):
        '''
        This function is used for inference by MINI BATCH(not training)
        Returns prediction and ground truth bbox in absolute coord.
        
        Params
        - loader: DataLoader
        - model: trained model
        - iou_threshold: iou threshold (used for NMS)
        - threshold: confidence score discard threshold
        
        return:
        - all_pred_boxes: predicted boxes by model
        - all_true_boxes: ground truth boxes
        '''
        all_pred_boxes = []
        all_true_boxes = []

        # make sure model is in eval before inferece
        model.eval()
        with torch.no_grad():
            train_idx = 0

            # inference loop
            for x, labels in loader:
                # to gpu
                x, labels = x.to(self.DEVICE), labels.to(self.DEVICE)

                # infer and normalize.
                predictions = model(x)
                preds_norm = self.box_processor.normalize(predictions)

                # convert cell box to absolute box
                batch_size = x.shape[0]
                preds_bboxes_list = self.box_processor.boxes_cell_to_list(preds_norm) # Convert prediction to absolute bbox.
                truth_bboxes_list = self.box_processor.boxes_cell_to_list(labels)

                # non-max-suppresion for each sample
                for idx in range(batch_size):
                    nms_boxes = Metric.non_max_suppression(
                        preds_bboxes_list[idx],
                        iou_threshold=iou_threshold,
                        threshold=threshold,
                        box_format=box_format,
                    )

                    for nms_box in nms_boxes:
                        all_pred_boxes.append([train_idx] + nms_box)

                    for box in truth_bboxes_list[idx]:
                        # many will get converted to 0 pred
                        if box[1] > threshold:
                            all_true_boxes.append([train_idx] + box)

                    train_idx += 1

        model.train()
        return all_pred_boxes, all_true_boxes

