import torch

class BoxProcesses:
    
    def __init__(self, S=7, B=2, C=20):
        self.S = S
        self.B = B
        self.C = C


    def normalize(self, raw_preds):
        '''
        This function normalizes the prediction within the value of 0~1
        
        Params
        - raw_preds: raw output from model

        Return
        - predictions: normalized prediction
        '''

        # Reshape output vector to 3d shape
        raw_preds = raw_preds.reshape(-1, self.S, self.S, self.C + self.B*5)

        # normalize bounding box
        # xy coordinate uses sigmoid in YOLOv2
        # wh size uses x^2 according to darknet github C code...
        box1_xy = raw_preds[..., 21:23]
        box1_wh = torch.square(raw_preds[..., 23:25])
        box2_xy = raw_preds[..., 26:28]
        box2_wh = torch.square(raw_preds[..., 28:30])
        # normalize confidence
        conf1_preds = raw_preds[..., 20:21] # box1
        conf2_preds = raw_preds[..., 25:26] # box2
        # normalize classification
        # class uses softmax in YOLOv2...
        class_preds = raw_preds[..., :20]

        # re-stack preds.
        predictions = torch.concat([class_preds, conf1_preds, box1_xy, box1_wh, conf2_preds, box2_xy, box2_wh], dim=-1)

        return predictions


    def boxes_absolute_to_cell(self, boxes):

        # convert boxes to label tensor
        label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            # now convert absolute x, y coordinate to cell coordinate
            # box w, y is already within [0, 1]
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            # this can label only one object to a grid cell if multiple objects overlap
            if label_matrix[i, j, 20] == 0:
                # label confidence = 1
                label_matrix[i, j, 20] = 1
                # label converted box coordinates
                box_coordinates = torch.tensor([x_cell, y_cell, width, height])
                label_matrix[i, j, 21:25] = box_coordinates
                # label class with one-hot encoding
                label_matrix[i, j, class_label] = 1 

        return label_matrix


    def _boxes_cell_to_absolute(self, predictions):
        """
        This function convert normalized predictions to absolute box shape.

        Params
        - predictions: normalized Tensor[batch_size, 7, 7, 30]

        Return
        - converted_preds: Tensor[batch_size, 7, 7, 6]
        """

        predictions = predictions.to("cpu")
        batch_size = predictions.shape[0]
        predictions = predictions.reshape(batch_size, 7, 7, 30)
        bboxes1 = predictions[..., 21:25]
        bboxes2 = predictions[..., 26:30]

        scores = torch.cat(
            (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0   # get Pc score
        )
        best_box = scores.argmax(0).unsqueeze(-1)                                           # choose one best_box for each cell.
        best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2                          # selects best_box
        
        # [..., [..., [[0], [1], [2], ..., [6]]]]
        # indices shape: [batch_size, 7, 7, 1]
        cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)               
        
        # (best_box center x) + (cell index) divided by 7 = absolute center x coord. (0~1)
        # (best_box center y) + (cell index) divided by 7 = absolute center y coord. (0~1)
        x = 1 / self.S * (best_boxes[..., :1] + cell_indices)
        y = 1 / self.S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))               # permute is just to reuse 0~6 for efficiency.
        w_xy = best_boxes[..., 2:4]

        # tensor shape:
        # [batch_size, 7, 7, 6]
        converted_bboxes = torch.cat((x, y, w_xy), dim=-1)
        predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
        best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
        converted_preds = torch.cat(
            (predicted_class, best_confidence, converted_bboxes), dim=-1
        )

        return converted_preds


    def boxes_cell_to_list(self, predictions):
        '''
        This function pulls all 49 boxes in the output and RETURN IN LIST TYPE.

        Params
        - predictions: SxS absolute bboxes. Tensor[batch_size, S, S, 30]
        
        Return
        - bboxes: List[batch_size, 49, 6] 
        '''

        absolute_boxes = self._boxes_cell_to_absolute(predictions)

        # reshape to (batch_size, 49, 6)
        converted_pred = absolute_boxes.reshape(absolute_boxes.shape[0], self.S * self.S, -1)
        # Convert class to "long" type
        converted_pred[..., 0] = converted_pred[..., 0].long()

        # for each batch
        all_bboxes = []
        for ex_idx in range(converted_pred.shape[0]):
            bboxes = []

            # collect bbox value from tensor.
            for bbox_idx in range(self.S * self.S):
                bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
            all_bboxes.append(bboxes)

        return all_bboxes