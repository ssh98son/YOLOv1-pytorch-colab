import os
import torch


import matplotlib.pyplot as plt
from tqdm import tqdm

from .box_processes import BoxProcesses
from .metric import Metric

class Trainer:
    
    def __init__(self, DEVICE):
        self.DEVICE = DEVICE

        # Training Records
        self.train_mean_losses = None
        self.val_mean_losses = None
        self.train_mAPs = None
        self.val_mAPs = None
        
        # defined when model is launched
        self.prev_epoch = None
        
        # defined when training is launched
        self.box_processor = None
        self.lr_scheduler = None


    def launch_model(self, model, optimizer, file_name, LOAD_MODEL=True):
        '''
        This function loads previous training data if exists.

        Params
        - model: model to train
        - optimizer: model optimizer
        - file_name: model save path
        - LOAD_MODEL: restart or load prev model. 
        '''
        
        # Load previous training if exists.
        if LOAD_MODEL and os.path.isfile(file_name):
            prev_train = self.load_checkpoint(model, optimizer, file_name)

            prev_epoch = prev_train[0]
            train_mean_losses = prev_train[1]
            val_mean_losses = prev_train[2]
            train_mAPs = prev_train[3]
            val_mAPs = prev_train[4]
            print(f"*** Continue previous training... Previous epoch was: {prev_epoch}***")
        else:
            prev_epoch = -1 # not 0
            train_mean_losses = []
            val_mean_losses = []
            train_mAPs = []
            val_mAPs = []
            print("*** Start training with new model... ***")


        # Set class fields
        self.train_mean_losses = train_mean_losses
        self.val_mean_losses = val_mean_losses
        self.train_mAPs = train_mAPs
        self.val_mAPs = val_mAPs
        self.prev_epoch = prev_epoch

        self.plot_training()
    
    
    def launch_training(self, optimizer):
        '''
        This function sets training environments.

        Params
        - optimizer: Optimizer to attach learning rate scheduler.
        '''
        
        # Validate launch_model
        if not self.prev_epoch and self.prev_epoch != -1:
            print("Model is not launched in trainer. Do launch_model() first.")
        
        # Init box processor, learning rate scheduler
        self.box_processor = BoxProcesses()
        self.lr_scheduler = self.LearningRateScheduler(optimizer, self.prev_epoch)


    def epoch_train_fn(self, train_loader, val_loader, model, optimizer, loss_fn):
        '''
        This function performs training for 1 epoch.
        
        Params
        - train_loader: DataLoader for train set
        - val_loader: DataLoader for validation set
        - model: Model to train
        - optimizer: Optimizer used for training
        - loss_fn: Loss function used for training

        Returns
        - train_mean_loss: mean training loss
        - val_mean_loss: mean validation loss
        '''    

        # ========================
        # Training Loop
        # ========================
        model.train()
        train_mean_loss = []
        loop = tqdm(train_loader)
        for (x, y) in loop:
            x, y = x.to(self.DEVICE), y.to(self.DEVICE)
            # feed forward
            out = model(x)
            # get loss (pass box processor for iou calculation)
            loss = loss_fn(out, y, self.box_processor)
            train_mean_loss.append(loss.item())
            # error back propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the progress bar
            loop.set_postfix(train_loss=loss.item())


        # ========================
        # Validation Loop
        # ========================
        model.eval()
        val_mean_loss = [] 
        with torch.no_grad():
            for (x, y) in val_loader:
                x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                # feed forward
                out = model(x)
                # get loss (pass box processor for iou calculation)
                val_loss = loss_fn(out, y, self.box_processor)
                val_mean_loss.append(val_loss.item())
        # change to train mode
        model.train()
        
        # ========================
        # Print Loss
        # ========================
        train_mean_loss = sum(train_mean_loss)/len(train_mean_loss)
        val_mean_loss = sum(val_mean_loss)/len(val_mean_loss)
        print(f"Mean train loss was      {train_mean_loss}")
        print(f"Mean validation loss was {val_mean_loss}")

        # update learning rate
        self.lr_scheduler.step()

        return train_mean_loss, val_mean_loss


    def check_mAP(self, data_loader, model, inferrer):
        '''
        Check mAP during training
        
        Params
        - model: YOLO model
        - data_loader: data loader for desired dataset
        - postprocessor: post processor for model output
        '''

        # toggle to evaluation mode
        model.eval()
        with torch.no_grad():
            # Check mAP on train set
            pred_boxes, target_boxes = inferrer.infer_labeled(
                data_loader, model, iou_threshold=0.5, threshold=0
            )
            mean_avg_prec = Metric.mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
        # toggle to training mode
        model.train()

        return mean_avg_prec


    def record_losses(self, train_mean_loss, val_mean_loss):
        self.train_mean_losses.append(train_mean_loss)
        self.val_mean_losses.append(val_mean_loss)


    def record_mAPs(self, train_mAP, val_mAP):
        self.train_mAPs.append(train_mAP)
        self.val_mAPs.append(val_mAP)


    def plot_training(self):

        epoch = len(self.train_mean_losses)
        plt.figure(figsize=(12,5))
        plt.subplot(1, 2, 1)
        plt.title("Losses")
        plt.plot(self.train_mean_losses, label="train")
        plt.plot(self.val_mean_losses, label="val")
        plt.xlabel("epoch")
        plt.xlim(left=0)
        plt.ylabel("loss")
        plt.ylim(bottom=0)
        plt.legend()

        if (len(self.train_mAPs)) != 0:
            step = int(epoch/len(self.train_mAPs))
            _x = [ 
                (v + 1) * step - 1 
                for v in range(len(self.train_mAPs))
            ]
        else:
            _x = []
        plt.subplot(1, 2, 2)
        plt.title("mAP")
        plt.plot(_x, self.train_mAPs, label="train")
        plt.plot(_x, self.val_mAPs, label="val")
        plt.xlabel("epoch")
        plt.xlim(left=0)
        plt.ylabel("mAP")
        plt.ylim(bottom=0)
        plt.legend()
        plt.show()


    def save_training(self, model, optimizer, epoch, filename):
        self.save_checkpoint(model, optimizer, epoch, 
                self.train_mean_losses, self.val_mean_losses,
                self.train_mAPs, self.val_mAPs,
                filename=filename)
        

    @staticmethod
    def save_checkpoint(model, optimizer, epoch,
                        train_mean_losses, val_mean_losses,
                        train_mAPs, val_mAPs,
                        filename="my_checkpoint.pth.tar"):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_mean_losses": train_mean_losses,
            "val_mean_losses": val_mean_losses,
            "train_mAPs": train_mAPs,
            "val_mAPs": val_mAPs,
        }
        torch.save(checkpoint, filename)

        print(f"=> Saving checkpoint: Epoch={checkpoint['epoch']}")


    @staticmethod
    def load_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar", ):
        checkpoint = torch.load(filename)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint['epoch']
        train_mean_losses = checkpoint["train_mean_losses"]
        val_mean_losses = checkpoint["val_mean_losses"]
        train_mAPs = checkpoint["train_mAPs"]
        val_mAPs = checkpoint["val_mAPs"]

        print("=> Loading checkpoint")

        return epoch, train_mean_losses, val_mean_losses, train_mAPs, val_mAPs


    class LearningRateScheduler(torch.optim.lr_scheduler.LambdaLR):
        '''
        Learning rate scheduler for training.
        - Warmups for 5 epochs
        - x 0.1 after 75 epochs
        - x 0.01 after 105 epochs
        '''
        def __init__(self, optimizer, last_epoch, warmup_steps=5):

            def lr_lambda(step):
                if step < warmup_steps:
                    factor = 0.1 + float(step)*0.9/float(warmup_steps)
                elif step < 75:
                    factor = 1.
                elif step < 105:
                    factor = 0.1
                else:
                    factor = 0.01

                return factor

            super().__init__(optimizer, lr_lambda, last_epoch=last_epoch, verbose=True)


