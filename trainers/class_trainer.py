import torch
import numpy as np

class ClassTrainer():
    def __init__(self, args, train_loader, device, optimizer, scaler, scheduler, criterion):
        self.args = args
        self.train_loader = train_loader
        self.device = device
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.criterion = criterion

    def train(self, model):
        model.train()
        total_train_loss = 0
        for i, data in enumerate(self.train_loader):
            images = data['img'].to(self.device)
            labels = data['class_label'].to(self.device)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(images)
                loss = self.criterion(pred, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_train_loss += loss.item()

        self.scheduler.step()

        train_loss = np.round(total_train_loss / len(self.train_loader), 2)
        print("TRAIN LOSS: " + str(train_loss))
        
        return train_loss
