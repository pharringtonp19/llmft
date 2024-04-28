import torch 
import numpy as np 
from dataclasses import dataclass, field

def no_op(*args, **kwargs):
    pass

# Example usage in your code
trace_func = no_op

@dataclass
class EarlyStopping:
    patience: int = 5
    verbose: bool = False
    delta: float = 0
    path: str = 'checkpoint.pt'
    trace_func: callable = no_op
    counter: int = field(default=0, init=False)
    val_loss_min: float = field(default=float('inf'), init=False)
    early_stop: bool = field(default=False, init=False)
    best_epoch: int = field(default=0, init=False)

    def __call__(self, val_loss, model, epoch):
        """Check if early stopping is needed and save model if improved."""
        if val_loss < self.val_loss_min - self.delta:
            self.val_loss_min = val_loss
            self.save_checkpoint(val_loss, model)
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                self.trace_func(f'Validation loss improved ({self.val_loss_min:.6f}). Saving model...')
        else:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves the model when validation loss decreases."""
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            self.trace_func(f'Model saved to {self.path} with validation loss {val_loss:.6f}')


@dataclass
class EncoderTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler 
    metric: any  # Define the type based on your implementation
    criterion:torch.nn.Module
    device: torch.device
    verbose: bool = True  # Default to True, can be set to False when creating an instance

    def train(self, data_loader):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []

        for batch in data_loader:
            loss, predictions, labels = self.process_batch(batch, train=True)
            total_loss += loss.item()
            all_predictions.append(predictions)
            all_labels.append(labels)

        return self.finalize_epoch(total_loss, all_predictions, all_labels, len(data_loader))

    def evaluate(self, data_loader):
        """Evaluate the model on a validation set."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                loss, _, _ = self.process_batch(batch, train=False)
                total_loss += loss.item()
        average_loss = total_loss / len(data_loader)
        return average_loss

    def process_batch(self, batch, train=True):
        """Process a single batch of data."""
        input_ids, attention_mask, labels = (batch['input_ids'].to(self.device), 
                                             batch['attention_mask'].to(self.device), 
                                             batch['labels'].to(self.device))
        logits = self.model(input_ids, attention_mask).logits
        loss = self.calculate_loss(logits, labels, batch)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        predictions = torch.argmax(logits, dim=1) if train else None
        return loss, predictions, labels

    def calculate_loss(self, logits, labels, batch):
        """Calculate loss based on the criterion and mode."""
        if self.criterion.mode == 'input' and 'type_indicator' in batch:
            type_indicator = batch['type_indicator'].to(self.device)
            return self.criterion(logits, labels, type_indicator)
        return self.criterion(logits, labels)

    def finalize_epoch(self, total_loss, all_predictions, all_labels, num_batches):
        """Finalize epoch, calculate metrics and average loss."""
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        metric_output = self.metric(all_predictions, all_labels)
        average_loss = total_loss / num_batches
        if self.verbose:
            print(f'Epoch finished. Average Loss: {average_loss:.4f}, Metric: {metric_output}')
        return average_loss, metric_output, self.scheduler.get_last_lr()[0]  # Assuming one parameter group

@dataclass
class DecoderTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler
    criterion: torch.nn.Module
    device: torch.device
    verbose: bool = True   
    gradient_accumulation: int = 8
    threshold: int = 10 

    def process_batch(self, batch):
        """Process a single batch of data, focusing only on the final token."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        logits = self.model(input_ids, attention_mask).logits
        labels = input_ids[:, -self.threshold:].contiguous()
        logits = logits[:, -self.threshold-1:-1, :].contiguous()
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss


    def train(self, data_loader):
        self.model.train()
        total_loss = 0
        total_samples = 0
        self.optimizer.zero_grad()

        for i, batch in enumerate(data_loader):
            loss = self.process_batch(batch)
            total_samples += len(batch)
            total_loss += loss.item() * len(batch)
            loss = loss / self.gradient_accumulation  # Normalizing loss for accumulation steps
            loss.backward()

            if (i + 1) % self.gradient_accumulation == 0:  # Optimizer step after 'accumulation_steps' batches
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                if self.verbose:
                    print(f"Batch {i+1} Loss: {loss.item() * self.gradient_accumulation}")
                torch.cuda.empty_cache()



        return total_loss / total_samples

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                loss = self.process_batch(batch).item()
                total_samples += len(batch)
                total_loss += loss * len(batch)

        return total_loss / total_samples




