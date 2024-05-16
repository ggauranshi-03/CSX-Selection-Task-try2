from models import CNNClassifier, save_model
from utils import ConfusionMatrix, load_data, LABEL_NAMES
import utils
import torch
import torchvision
import torch.utils.tensorboard as tb
from os import path

def train(args):
    #model
    model = CNNClassifier()
    #init tb writer
    train_logger, valid_logger = None, None

    if args.log_dir is not None:

        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th')))

    #to gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    
    # setting optim and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=0.0001) 
    loss = torch.nn.CrossEntropyLoss()

    #loading data
    train_data = load_data('data/train', transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        torchvision.transforms.RandomAffine(degrees=(-15, 15), translate=(0.2, 0.2), scale=(0.8, 1.2)),  
        torchvision.transforms.ToTensor(),
    ]))

    valid_loader = load_data('data/valid', transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),

    ])) 


    
    
    
    global_step = 0
    #train loop
    for epoch in range(args.num_epoch):
        model.train()
        cm = ConfusionMatrix(size=len(LABEL_NAMES)) # Initialize ConfusionMatrix

        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            logit = model(img)
            loss_val = loss(logit, label)
            acc_val = accuracy(logit, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            cm.add(logit.argmax(1), label) # Update ConfusionMatrix

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

            # Logging with TensorBoard
            if train_logger:
                
                train_logger.add_scalar('loss', loss_val, global_step)
                train_logger.add_scalar('accuracy', cm.global_accuracy, global_step) 

        # Validation
        model.eval()
        accuracies = []
        with torch.no_grad():
            for img, label in valid_loader:
                img, label = img.to(device), label.to(device)
                accuracies.append(accuracy(model(img), label).item())

        avg_accuracy = sum(accuracies) / len(accuracies)
        if valid_logger:
            valid_logger.add_scalar('accuracy', avg_accuracy, global_step)

        # Print & save 
        
        print(f'epoch {epoch}, train acc: {acc_val:.3f}, val acc: {avg_accuracy:.3f}') 
        save_model(model)
    save_model(model)
       
def accuracy(output, target):
    """Computes the accuracy for multi-class classification"""
    return (output.argmax(1) == target).float().mean()
    """
    You might need to tweak or add to the above code to make the CNN classifier run
    """


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('--continue_training', default=False)
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--num_epoch', default=120)

    # Put custom arguments here

    args = parser.parse_args()
    train(args)
