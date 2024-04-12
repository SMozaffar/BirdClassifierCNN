from models import CNNClassifier, save_model, load_model
from utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import os


def train(args):
    from os import path

    # Load or create the model
    if args.continue_training and os.path.exists(args.model_path):
        model = load_model()
    else:
        model = CNNClassifier()
    
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Load data
    train_loader = load_data(args.train_dataset_path, num_workers=args.num_workers, batch_size=args.batch_size, split='train')
    valid_loader = load_data(args.valid_dataset_path, num_workers=args.num_workers, batch_size=args.batch_size, split='valid')
    
    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0
        for i, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            o = model(data)
            loss = criterion(o, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += accuracy(o, targets).item()

            # Log training loss
            train_logger.add_scalar('loss', loss.item(), epoch * len(train_loader) + i)

        # Log training accuracy
        avg_train_accuracy = train_accuracy / len(train_loader)
        train_logger.add_scalar('accuracy', avg_train_accuracy, epoch)

        # Validation 
        model.eval()
        valid_accuracy = 0
        with torch.no_grad():
            for data, targets in valid_loader:
                outputs = model(data)
                valid_accuracy += accuracy(outputs, targets).item()

        # Log validation accuracy
        avg_valid_accuracy = valid_accuracy / len(valid_loader)
        valid_logger.add_scalar('accuracy', avg_valid_accuracy, epoch)

        
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True)  
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('-tdp', '--train_dataset_path', type=str, required=True)
    parser.add_argument('-vdp', '--valid_dataset_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='cnn.th')
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()
    train(args)
