import wandb
import torch
import numpy as np

class RegressionEvaluator():
    def __init__(self, args, test_loader, device, criterion, run_name, metric):
        self.args = args
        self.test_loader = test_loader
        self.device = device
        self.criterion = criterion
        self.best_error = 1000
        self.run_name = run_name
        self.metric = metric

    def test(self, model):
        model.eval()
        total_val_loss = 0
        total = 0
        total_error = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                images = data['img'].to(self.device)
                labels = data['label'].to(self.device)
                labels = labels.unsqueeze(1)

                pred = model(images)
                if 'norm' in self.args.model:
                    pred = pred * 360

                loss = self.criterion(pred, labels)
                total_val_loss += loss.item()

                total_error += self.metric(pred, labels)
                total += labels.size(0)
        
        test_loss = np.round(total_val_loss / len(self.test_loader), 2)
        total_error = total_error.item()
        cmae = np.round(total_error / total, 2)
        print("Test Loss: " + str(test_loss))
        print("CMAE: " + str(cmae) + " Best CMAE: " + str(self.best_error))
        if cmae < self.best_error:
            self.best_error = cmae
            # save checkpoint
            torch.save(model.state_dict(), 'saved_models/' + self.run_name + '.pth')
            print("Saved best model")
        
        print("=" * 100)

        return test_loss, cmae