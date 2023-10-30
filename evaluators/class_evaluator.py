import wandb
import torch
import numpy as np

class ClassEvaluator():
    def __init__(self, args, test_loader, device, criterion, run_name, metric):
        self.args = args
        self.test_loader = test_loader
        self.device = device
        self.criterion = criterion
        self.best_accuracy = 0
        self.run_name = run_name
        self.metric = metric

    def test(self, model):
        model.eval()
        total_val_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                images = data['img'].to(self.device)
                labels = data['class_label'].to(self.device)

                pred = model(images)
                loss = self.criterion(pred, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(pred, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        test_loss = np.round(total_val_loss / len(self.test_loader), 2)
        total_correct = correct
        accuracy = np.round(100 * total_correct / total, 2)

        print("Test Loss: " + str(test_loss))
        print("Accuracy: " + str(accuracy) + " Best Accuracy: " + str(self.best_accuracy))
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            # save checkpoint
            torch.save(model.state_dict(), 'saved_models/' + self.run_name + '.pth')
            print("Saved best model")
        
        print("=" * 100)

        return test_loss, accuracy