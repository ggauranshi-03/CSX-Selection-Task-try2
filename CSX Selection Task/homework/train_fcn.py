import torch, torchvision
import numpy as np

from models import FCN, save_model
from utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix, DenseSuperTuxDataset
import dense_transforms
import torch.utils.tensorboard as tb



def train(args):
    from os import path

    # TensorBoard Setup 
    model = FCN()  
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Data Loading + Augmentations
    transform = dense_transforms.Compose([
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ToTensor(),   
    ])
    #dataset = DenseSuperTuxDataset('dense_data/dense_data/train', transform=transform)  
    # dataloader = load_dense_data('dense_data/train', batch_size=32, num_workers=4)  
    train_data = load_dense_data('**/train', transform=transform)

    valid_loader = load_dense_data('**/valid', transform=transform)


    # Model, Loss, Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Weighted Loss Calculation
    class_weights = torch.FloatTensor(DENSE_CLASS_DISTRIBUTION).to(device) 
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights) 

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=0.0001)   

    # Training Loop
    for epoch in range(args.num_epochs):
        
        model.train()  
        cm = ConfusionMatrix(size=5)  
        i = 0
        acc_val = 0
        iou = 0
        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            label = label.type(torch.long)
            logits = model(img)
            loss = loss_function(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            cm.add(logits.argmax(1), label) 

            acc_val = cm.global_accuracy
            iou = cm.iou
            print(f"Step {i}, global acc: {acc_val:.3f}, iou: {iou:}")

            i+=1

        # Logging 
        if train_logger:
            train_logger.add_scalar('loss', loss, epoch)
            train_logger.add_scalar('iou', cm.iou, epoch)
        
        # model.eval()
        # valid_global_accuracy = 
        # if valid_logger:
        #     valid_logger.add_scalar('accuracy', avg_accuracy, global_step)
        
        print(f'epoch {epoch}') 

    # Model Saving
    save_model(model)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default = './logs2')
    parser.add_argument('--num_epochs', default = 3)
    parser.add_argument('--learning_rate', default = 0.002)

    # Put custom arguments here

    args = parser.parse_args()
    train(args)

