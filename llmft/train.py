import torch 
import numpy as np 

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

def train_one_epoch(model, train_loader, optimizer, scheduler, metric, criterion, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        logits = outputs.logits  # Extract logits from the model output
        
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



