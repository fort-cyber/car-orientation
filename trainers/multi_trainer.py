import torch
import numpy as np

class MultiTrainer():
    def __init__(self, args, train_loader, device, optimizer, scaler, scheduler, criterion):
        self.args = args
        self.train_loader = train_loader
        self.device = device
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.class_criterion, self.regression_criterion = criterion
        self.alpha = 10

    def train(self, model):
        model.train()
        total_train_loss = 0
        for i, data in enumerate(self.train_loader):
            images = data['img'].to(self.device)
            class_labels = data['class_label'].to(self.device)
            regression_labels = data['label'].to(self.device)
            regression_labels = regression_labels.unsqueeze(1)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                class_pred, angle_pred = model(images)
                if 'norm' in self.args.model:
                    angle_pred = angle_pred * 360

                class_loss = self.alpha * self.class_criterion(class_pred, class_labels)
                regression_loss = self.regression_criterion(angle_pred, regression_labels)
                loss = class_loss + regression_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_train_loss += loss.item()

        self.scheduler.step()

        train_loss = np.round(total_train_loss / len(self.train_loader), 2)
        print("TRAIN LOSS: " + str(train_loss))
        
        return train_loss
