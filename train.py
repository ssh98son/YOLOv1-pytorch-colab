import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader

from importlib import reload
import sys
import os.path 
sys.path.append("/content/drive/MyDrive/Colab Resources/YOLOv1")

from yolo.resnet_model_light import ResnetYoloV1
from yolo.loss import YoloLoss
from yolo.dataset import VOCDataset
from utils.metric import Metric
from utils.image_processes import ResizePreprocess, JitterPreprocess, normalize_imagenet_to_cv2
from utils.box_processes import BoxProcesses
from utils.trainer import Trainer
from utils.inferer import Inferer
from utils.visualize import draw_box_on_image


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Local Hyperparameters etc.
NUM_WORKERS = 2
PIN_MEMORY = True

# Tunable Hyperparameters (in Colab)
BATCH_SIZE = 64
LEARNING_RATE = 0.005
EPOCHS = 135
WEIGHT_DECAY = 0.0005
MODEL_CKPT_PATH = "/content/drive/MyDrive/Colab Resources/YOLOv1/resnet_model.pth.tar"


if __name__ == "__main__":
    # =============
    # Load 
    # =============
    train_dataset = VOCDataset(
        csv_file="/content/PascalVOC/train.csv",
        transform=JitterPreprocess(),
        img_dir="/content/PascalVOC/images",
        label_dir="/content/PascalVOC/labels"
    )

    val_dataset = VOCDataset(
        csv_file="/content/PascalVOC/test.csv",
        transform=ResizePreprocess(),
        img_dir="/content/PascalVOC/images",
        label_dir="/content/PascalVOC/labels"
    )


    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False
    )

    # ======
    # Initialize Model
    # ======
    model = ResnetYoloV1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()


    # setup trainer
    trainer = Trainer(DEVICE)
    # load previous training
    trainer.launch_model(model, optimizer, MODEL_CKPT_PATH, LOAD_MODEL=True)
    # set learning rate scheduler
    trainer.launch_training(optimizer)

    # setup inferer
    inferer = Inferer(DEVICE)


    # =====
    # Epoch Loop
    # =====
    for epoch in range(trainer.prev_epoch+1, EPOCHS):
        print(f"Epoch: {epoch}")

        # Training
        train_mean_loss, val_mean_loss = trainer.epoch_train_fn(train_loader, val_loader, model, optimizer, loss_fn)
        # Record losses
        trainer.record_losses(train_mean_loss, val_mean_loss)
        
        if (epoch+1) % 1 == 0:
            
            if (epoch+1) % 5 == 0:
                # Check mAP performance
                train_mAP = 0
                val_mAP = trainer.check_mAP(val_loader)
                print(f"Train mAP: {train_mAP}, Validation mAP: {val_mAP}")
                # Record mAPs
                trainer.record_mAPs(train_mAP, val_mAP)

                # Plot record
                trainer.plot_training()

            # Save checkpoint during training
            trainer.save_training(model, optimizer, epoch, filename=MODEL_CKPT_PATH)
        

