import torch
import torch.nn as nn
from utils.metric import Metric

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.sse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5 # cost for false-false confidence regression
        self.lambda_coord = 5 # cost for true-true box regression
        self.epsilon = 1e-7


    def forward(self, predictions, target, boxprocessor):
        '''
        Params
        - predictions: tensor(N, S, S, 30), normalized.
        - target: tensor (N, S, S, 25)
        where N is size of MINI-BATCH.

        Return
        - loss: tensor(N, 1)

        Prediction vector: [c1, c2, ..., c20, pc1, x, y, w, h, pc2, x, y, w, h]
        Target label vector: [c1, c2, ..., c20, pc, x, y, w, h]
        '''

        # Reshape output vector to 3d shape
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)

        # prediction [..., 21:25] contains bounding box1 for all grid cells
        # prediction [..., 26:30] contains bounding box2 for all grid cells
        # normalized prediction will be used to calculate iou. (w_actual = w_pred ** 2)
        predictions_norm = boxprocessor.normalize(predictions.clone().detach())
        iou_b1 = Metric.intersection_over_union(predictions_norm[..., 21:25], target[..., 21:25])
        iou_b2 = Metric.intersection_over_union(predictions_norm[..., 26:30], target[..., 21:25])
        
        # find one responsible box for each cell.
        # unsqeezing ious converts shape (N, S, S, 1) * 2 to (2, N, S, S, 1)
        # torch.max pulls iou_max and best_box indice per grid. (2, N, S, S, 1) to (N, S, S, 1) * 2
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_max, best_box = torch.max(ious, dim=0)

        # exists_box is "Iobj_i" in the paper. (used for confidence, Pc label)
        # change shape to (N, S, S) -> (N, S, S, 1) (scalar to array size of 1)
        exists_box = target[..., 20].unsqueeze(3)

        # ===============================
        # Box Location & Shape Loss
        # ===============================
        # best_box pulls b1 or b2 with better iou.
        # exists_box selects responsible grid cells.
        
        # pull boxes of responsible predictions
        box_predictions = exists_box * (
            (
                best_box * predictions[..., 26:30]
                + (1 - best_box) * predictions[..., 21:25]
            )
        )
        # pull boxes of label
        box_targets = exists_box * target[..., 21:25]

        # take sqrt of box w, h
        # add epsilon for stability

        # tensor shape (N, S, S, 4) -> (N, S, S)
        # box location loss
        box_loss = torch.sum(
            torch.square(
                box_predictions[..., 0:2]
                - box_targets[..., 0:2]
            ), 
            dim=-1
        )
        # box shape loss
        box_loss += torch.sum(
            torch.square(
                box_predictions[..., 2:4] # box prediction logits are sqrt of w, h
                - torch.sqrt(box_targets[..., 2:4])
            ),
            dim=-1
        )

        # ===============================
        # Object Loss (Confidence Score)
        # Penalizing when object exists(pc=1), but model is not confident.
        # ===============================
        # best_box pulls responsible prediction. b1 or b2
        pred_confidence = (
            best_box * predictions[..., 25:26]
            + (1 - best_box) * predictions[..., 20:21]
        )

        # target Confidence Score = 1 * IOU^truth_pred
        # scalar multiplication: (N, S, S, 1) * (N, S, S, 1) 
        target_confidence = target[..., 20:21] * iou_max

        # flatten tensor shape (N, S, S, 1) -> (N, S, S)
        # exists_box pulls responsible grid cells.
        object_loss = torch.sum(
            torch.square(
                (exists_box * pred_confidence)
                - (exists_box * target_confidence)
            ),
            dim=-1
        )

        # ===============================
        # No Object Loss
        # Penalizing when object doesn't exists(pc=0), but model predicts exists.
        # ===============================
        # (1-exists_box) pulls not responsible grid cells.
        
        
        # tensor shape (N, S, S, 1) -> (N, S, S)
        # box1
        no_object_loss = torch.sum(
            torch.square(
                (1 - exists_box) * predictions[..., 20:21]
                - (1 - exists_box) * target[..., 20:21]
            ),
            dim=-1
        )
        # box2
        no_object_loss += torch.sum(
            torch.square(
                (1 - exists_box) * predictions[..., 25:26]
                - (1 - exists_box) * target[..., 20:21]
            ),
            dim=-1
        )

        # ===============================
        # Classification Loss
        # Wrong classification when object exists
        # ===============================      
        # class tensor shape (N, S, S, 20) -> (N, S, S)
        class_loss = torch.sum(
            torch.square(
                exists_box * predictions[..., :20]
                - exists_box * target[..., :20]
            ),
            dim=-1
        )

        # ===============================
        # loss tenor shape (N, S, S) -> (N)
        loss = torch.sum(
            self.lambda_coord * box_loss 
            + object_loss 
            + self.lambda_noobj * no_object_loss
            + class_loss
            , dim=[1, -1]
        )

        # batch loss mean reduction
        return torch.mean(loss)


