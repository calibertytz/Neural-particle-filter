import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from filterdata_loader import FilterDataset
from RNN import lstm
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

default_configs = {'input_size': 2,
                   'hidden_size': 16,
                   'output_size': 1,
                   'num_layers ': 2,
                   'batch_size': 100,
                   'learning_rate': 0.01,
                   'threshold': 1e-3,
                   'num_epochs': 10}

root_dir = ""
train_feature = os.path.join(root_dir, 'x_train.npy')
train_label = os.path.join(root_dir, 'y_train.npy')
val_feature = os.path.join(root_dir, "x_test.npy")
val_label = os.path.join(root_dir, 'y_test.npy')

train_data = FilterDataset(feature_path=train_feature, label_path=train_label)
val_data = FilterDataset(feature_path=val_feature, label_path=val_label)

train_loader = DataLoader(train_data, batch_size=default_configs['batch_size'], shuffle=False, num_workers=4)
test_loader = DataLoader(val_data, batch_size=default_configs['batch_size'], shuffle=False, num_workers=4)


model = lstm().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=default_configs['learning_rate'])

def train():
    # Train the model
    print('model is training:')
    total_step = len(train_loader)
    flag = True
    for epoch in range(default_configs['num_epochs']):
        for i, (x, y) in enumerate(train_loader):
            var_x = x.to(device)
            var_y = y.to(device)
            # forward pass
            out = model(var_x)
            loss = criterion(out, var_y)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss.item() <= default_configs['threshold']:
                flag = False
                break
            if (i + 1) % 1 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, default_configs['num_epochs'], i + 1, total_step, loss.item()))
                val(epoch)
        if not flag:
            break

def val(epoch):
    print('model is evaluating:')
    model.eval()  # converting to test model
    with torch.no_grad():
        total_val_loss, count = 0, 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred_test = model(x).cpu()
            count += 1
            total_val_loss += criterion(pred_test, y)
        print('Average val loss {: .4f}'.format(total_val_loss/count))

if __name__  == '__main__':
    val(0)
    train()
    torch.save(model.state_dict(), 'model_NF.pth')





