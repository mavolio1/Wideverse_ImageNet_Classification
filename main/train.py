import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import dataset
from model import custom_net
from config import cfg

#
#   Main training loop for the model.
#   All the hyperparameter are defined in the config module.
#
def main():
    # Loss function, optimizer and learning rate scheduler definition
    model = custom_net.get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model.train()

    # Preparation of the DataLoader with the Data Augumentation
    preprocess = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip()
    ])
    # The lambda function is used to change the range of labels from [1, 4] -> [0, 3]
    training_data = dataset.ImageNet_Challenge_Dataset(cfg.DATA_DIR + 'train.csv', cfg.DATA_DIR + 'train', transform=preprocess, target_transform=lambda x : x-1)
    train_dataloader = DataLoader(training_data, batch_size=cfg.batch_size, shuffle=True)

    since = time.time()
    best_acc = 0.0
    best_epoch = 0

    # Main training loop
    for idx, epoch in enumerate(range(cfg.epochs)):
        print(f'Epoch {epoch + 1}/{cfg.epochs}')
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_dataloader):
            # Move data to device for optimization
            inputs = inputs.to(cfg.device)
            labels = labels.to(cfg.device)

            # Zero the gradients before updating them
            optimizer.zero_grad()

            # Forward step
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward step
            loss.backward()
            optimizer.step()

            # Metric calculation
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)


        scheduler.step()
        epoch_loss = running_loss / len(training_data)
        epoch_acc = running_corrects.double() / len(training_data)

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Save all snapshot, for longer training session or bigger models, we can save only every x epochs
        torch.save(model.state_dict(), cfg.CHECKPOINT_DIR + 'snapshot' + str(epoch) + '.pt')

        # Save the best Accuracy just to show at the end of the training session
        if epoch_acc > best_acc:
            best_acc = epoch_acc

    print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f} in epoch: {best_epoch}')


if __name__ == '__main__':
    main()