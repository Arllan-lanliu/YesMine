import sys
import torch
import numpy as np
import os

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path


from ..utils.util import get_model_save_related
from ..data.dataloader import get_dataloader


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, use_logits=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.use_logits = use_logits
        
    def forward(self, logits, targets):
        if self.use_logits:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()
    
def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    correct=0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight = weight)
    num_batch = len(dev_loader)
    i=0
    with torch.no_grad():
      for batch_x, batch_y in dev_loader:
        batch_size = batch_x.size(0)
        target = torch.LongTensor(batch_y).to(device)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)

        # batch_y = batch_y.to(device)
        # batch_y = batch_y.float() * 0.9 + 0.05
        # correct += batch_out.eq(target).sum().item()
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        pred = batch_out.max(1)[1] 
        correct += pred.eq(target).sum().item()

        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)

        i=i+1
        print("batch %i of %i (Memory: %.2f of %.2f GiB reserved) (validation)"
                  % (
                     i,
                     num_batch,
                     torch.cuda.max_memory_allocated(device) / (2 ** 30),
                     torch.cuda.max_memory_reserved(device) / (2 ** 30),
                     ),
                  end="\r",
                  )

    val_loss /= num_total
    test_accuracy = 100. * correct / len(dev_loader.dataset)
    print('Test accuracy: ' +str(test_accuracy)+'%')
    return val_loss


def train_epoch(train_loader, model, lr, optimizer, criterion, device):
    num_total = 0.0
    model.train()

    num_batch = len(train_loader)
    i=0
    pbar = tqdm(train_loader, total = num_batch)
    for batch_x, batch_y in pbar:
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_out = model(batch_x)

        # batch_y = batch_y.to(device)
        # batch_y = batch_y.float() * 0.9 + 0.05
        # batch_loss = criterion(batch_out, batch_y)

        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_loss = criterion(batch_out, batch_y)          

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        i = i+1

    sys.stdout.flush()


def train_model(config, config_path, model, device):
    print('######## Training ########')

    model_save_path, best_save_path, model_tag = get_model_save_related(config)
    print(f'Model tag: {model_tag}')
    
    train_dataloader = get_dataloader(config, subset = 'train')
    dev_dataloader = get_dataloader(config, subset = 'dev')
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr = config.lr, 
                                weight_decay = config.weight_decay)
    
    #criterion = BinaryFocalLoss()

    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight = weight)

    not_improving = 0
    epoch = 0
    n_mejores = config.n_mejores_loss
    bests = np.ones(n_mejores, dtype = float) * float('inf')
    best_loss = float('inf')

    # Initialize best model files
    for i in range(n_mejores):
        file_path = Path(os.path.join(best_save_path, f'best_{i}.pth'))
        file_path.parent.mkdir(parents = True, exist_ok = True)
        np.savetxt(file_path, np.array((0, 0)))
    
    best_path = Path(os.path.join(model_save_path, 'best.pth'))
    best_path.parent.mkdir(parents = True, exist_ok = True)
    os.system(f'cp {config_path} {model_save_path}')
    np.savetxt(best_path, np.array((0, 0)))

    while not_improving < config.impove_epoch_patience:
        print(f'######## Epoch {epoch} ########')
        
        train_epoch(train_dataloader, model, config.lr, optimizer, criterion, device)
        val_loss = evaluate_accuracy(dev_dataloader, model, device)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best.pth'))
            print('New best epoch')
            not_improving = 0
        else:
            not_improving += 1
        
        # Update n-best models
        for i in range(n_mejores):
            if bests[i] > val_loss:
                for t in range(n_mejores - 1, i, -1):
                    bests[t] = bests[t - 1]
                    os.system(f'mv {best_save_path}/best_{t - 1}.pth {best_save_path}/best_{t}.pth')
                bests[i] = val_loss
                torch.save(model.state_dict(), os.path.join(best_save_path, f'best_{i}.pth'))
                break
        
        print(f'\n{epoch} - {val_loss}')
        print(f'n-best loss: {bests}')
        epoch += 1
        
        if epoch > config.train_epoch:
            break
             
    print(f'Total epochs: {epoch}\n')