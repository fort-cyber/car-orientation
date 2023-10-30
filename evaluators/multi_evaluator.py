import wandb
import torch
import numpy as np

class MultiEvaluator():
    def __init__(self, args, test_loader, device, criterion, run_name, metric):
        self.args = args
        self.test_loader = test_loader
        self.device = device
        self.class_criterion, self.regression_criterion = criterion
        self.best_error = 1000
        self.best_accuracy = 0
        self.run_name = run_name
        self.metric = metric

    def test(self, model):
        model.eval()
        total_val_loss = 0
        total = 0
        total_regression_error = 0
        correct = 0

        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                images = data['img'].to(self.device)
                class_labels = data['class_label'].to(self.device)
                regression_labels = data['label'].to(self.device)
                regression_labels = regression_labels.unsqueeze(1)

                class_pred, angle_pred = model(images)
                if 'norm' in self.args.model:
                    angle_pred = angle_pred * 360

                class_loss = self.class_criterion(class_pred, class_labels)
                regression_loss = self.regression_criterion(angle_pred, regression_labels)
                loss = class_loss + regression_loss
                total_val_loss += loss.item()

                total_regression_error += self.metric(angle_pred, regression_labels)

                _, predicted = torch.max(class_pred, dim=1)
                correct += (predicted == class_labels).sum().item()
                total += class_labels.size(0)
        
        test_loss = np.round(total_val_loss / len(self.test_loader), 2)
        total_regression_error = total_regression_error.item()
        cmae = np.round(total_regression_error / total, 2)
        total_correct = correct
        accuracy = np.round(100 * total_correct / total, 2)

        print("Test Loss: " + str(test_loss))
        print("CMAE: " + str(cmae) + " Best CMAE: " + str(self.best_error))
        print("Accuracy: " + str(accuracy) + " Best Accuracy: " + str(self.best_accuracy))

        if cmae < self.best_error:
            self.best_error = cmae
            # save checkpoint
            torch.save(model.state_dict(), 'saved_models/' + self.run_name + '.pth')
            print("Saved best model")

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy 

        print("=" * 100)

        return test_loss, accuracy, cmae