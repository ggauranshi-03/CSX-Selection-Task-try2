from os import path
import torch
import torch.utils.tensorboard as tb
import numpy as np

def test_logging(train_logger, valid_logger):

    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """

    # This is a strongly simplified training loop
    train_acc = []
    val_acc = []
    train_loss = []
    for epoch in range(10):
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9 ** (epoch + iteration / 20.)
            dummy_train_accuracy = epoch / 10. + torch.randn(10)
            train_accuracy = torch.mean(dummy_train_accuracy)
            global_step = 0
            train_logger.add_scalar('Training Loss', dummy_train_loss, global_step=0)
            #train_logger.add_scalar('Training Accuracy', train_accuracy, global_step=epoch)

        train_accuracy = train_accuracy.tolist()

        train_acc.append(train_accuracy)
        accuracy = np.mean(train_acc)
        train_logger.add_scalar('Training Accuracy', train_accuracy, global_step=epoch)
        torch.manual_seed(epoch)
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
        val_accuracy = torch.mean(dummy_validation_accuracy)
        val_accuracy = val_accuracy.tolist()
        val_acc.append(val_accuracy)
        accuracy = np.mean(val_acc)
        valid_logger.add_scalar('Validation Accuracy', accuracy, global_step=1)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
