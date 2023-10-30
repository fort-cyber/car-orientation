import torch
import torch.optim
import torch.utils.data
from common import *
from utils import *
import wandb
from trainers import *
from evaluators import *

# Arguments
args = define_args()
config = vars(args)

# Set manual seed for reproducibility
set_seed(args.seed)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging
if not args.run_name:
	run_name = args.model + '_' + str(args.exp_index)
else:
	run_name = args.run_name

if args.wandb_log:
	project_name = 'car-angle'
	wandb.init(project=project_name)
	wandb.run.name = run_name
	wandb.config.update({'config': config})

# Hyperparameters
workers = args.num_workers
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
train_loader, test_loader = get_loaders(args)

# Use mixed precision
scaler = torch.cuda.amp.GradScaler()
torch.backends.cudnn.benchmark = True

# Model
model = get_model(args)
model.to(device)

# Training helpers
criterion = get_criterion(args)
metric = CircularDistance()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = get_scheduler(optimizer, args, len(train_loader))

print(run_name)

# Trainer and Evaluator
trainer = get_trainer(args, train_loader, device, optimizer, scaler, scheduler, criterion)
evaluator = get_evaluator(args, test_loader, device, criterion, run_name, metric)

# Training loop
for epoch in range(epochs):
	print("EPOCH " + str(epoch))

	train_loss = trainer.train(model)
	current_lr = trainer.scheduler.get_last_lr()[0]
	
	if args.trainer == 'regression':
		test_loss, cmae = evaluator.test(model)
		if args.wandb_log:
			wandb.log({
				"Test loss": test_loss,
				"Train loss": train_loss,
				"CMAE": cmae,
				"Learning Rate": current_lr,
			}, step=epoch)

	elif args.trainer == 'classification':
		test_loss, accuracy = evaluator.test(model)
		if args.wandb_log:
			wandb.log({
				"Test loss": test_loss,
				"Train loss": train_loss,
				"Accuracy": accuracy,
				"Learning Rate": current_lr,
			})
	
	elif args.trainer == 'multi':
		test_loss, accuracy, cmae = evaluator.test(model)
		if args.wandb_log:
			wandb.log({
				"Test loss": test_loss,
				"Train loss": train_loss,
				"CMAE": cmae,
				"Accuracy": accuracy,
				"Learning Rate": current_lr,
			})

wandb.finish()