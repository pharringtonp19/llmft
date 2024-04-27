import torch 
import numpy as np 
from dataclasses import dataclass

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.val_loss_min = float('inf')
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_epoch = 0  # Attribute to track the epoch number of the best validation loss

    def __call__(self, val_loss, model, epoch):
        if self.val_loss_min == float('inf') or val_loss < self.val_loss_min - self.delta:
            if val_loss < self.val_loss_min:
                self.val_loss_min = val_loss
                self.save_checkpoint(val_loss, model)
                self.best_epoch = epoch  # Update the best epoch
                self.counter = 0
            if self.verbose:
                self.trace_func(f'Validation loss improved ({self.val_loss_min:.6f}). Saving model...')
        else:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.path)


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits  # Extract logits from the model output
            # Handling for the type_indicator based on the mode
            
            if criterion.mode == 'input':
                type_indicator = batch['type_indicator'].to(device)   
                loss = criterion(logits, labels, type_indicator)
            else:
                type_indicator = None
                loss = criterion(logits, labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def train_encoder(model, train_loader, optimizer, scheduler, metric, criterion, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        logits = model(input_ids, attention_mask).logits
        
        # Handling for the type_indicator based on the mode
        if criterion.mode == 'input':
            type_indicator = batch['type_indicator'].to(device)  # Ensure your dataloader provides this if needed
            loss = criterion(logits, labels, type_indicator)
        else:
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()

        # Accumulate all predictions and labels for F1 score calculation
        predictions = torch.argmax(logits, dim=1)  # Convert logits to predicted class indices
        all_predictions.append(predictions)
        all_labels.append(labels)

    # Concatenate all predictions and labels across batches
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    # Compute the weighted F1 score
    metric_output = metric(all_predictions, all_labels)

    average_loss = total_loss / len(train_loader)

    return average_loss, metric_output, scheduler.get_last_lr()[0]  # Assuming you have only one parameter group


@dataclass
class ModelTrainer:
    model: torch.nn.Module
    train_loader: torch.utils.data.DataLoader
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler
    metric: any  # Define the type based on your implementation
    criterion: torch.nn.Module
    device: torch.device
    verbose: bool = True  # Default to True, can be set to False when creating an instance

    def train_decoder(self):
        self.model.train()
        total_loss = 0

        accumulation_steps = 8  # Adjust based on memory capacity and desired effective batch size
        self.optimizer.zero_grad()
        for i, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            labels = input_ids[:, 1:].clone().detach()
            input_ids = input_ids[:, :-1]

            attention_mask = batch['attention_mask'].to(self.device)
            attention_mask = attention_mask[:, :-1]

            logits = self.model(input_ids, attention_mask).loss['logits']
            mask = labels != -100
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = (loss * mask.view(-1)).sum() / mask.sum()

            loss = loss / accumulation_steps  # Normalize loss to account for accumulation
            loss.backward()
            total_loss += loss.item()
            if (i + 1) % accumulation_steps == 0:  # Perform optimization step after 'accumulation_steps' batches
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                if self.verbose:
                    print(f"Batch Loss: {loss.item()}")
                torch.cuda.empty_cache()

        average_loss = total_loss / len(self.train_loader)

        return average_loss



