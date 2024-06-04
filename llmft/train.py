import torch 
import numpy as np 
from dataclasses import dataclass, field
from transformers import PreTrainedTokenizerBase


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
        criterion_mode = getattr(self.criterion, 'mode', None)  # Defaulting to None or another appropriate default
        if criterion_mode == 'input' and 'type_indicator' in batch:
            type_indicator = batch['type_indicator'].to(self.device)
            loss = self.criterion(logits, labels, type_indicator)
        else:
            loss = self.criterion(logits, labels)
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        predictions = torch.argmax(logits, dim=1) if train else None
        return loss, predictions, labels

    def finalize_epoch(self, total_loss, all_predictions, all_labels, num_batches):
        """Finalize epoch, calculate metrics and average loss."""
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        metric_output = self.metric(all_predictions, all_labels)
        average_loss = total_loss / num_batches
        if self.verbose:
            print(f'Epoch finished. Average Loss: {average_loss:.4f}, Metric: {metric_output}')
        return average_loss, metric_output, self.scheduler.get_last_lr()[0]  

@dataclass
class DecoderTrainer:
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler
    criterion: torch.nn.Module
    device: torch.device
    verbose: bool = True   
    gradient_accumulation: int = 8
    threshold: int = 10 

    def process_batch(self, batch, token_type = 'target_token'):
        """Compute logits and labels for a batch of data."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        logits = self.model(input_ids, attention_mask).logits
        return input_ids, logits, batch['type_indicator'].to(self.device), batch[token_type].to(self.device)

    def compute_loss(self, input_ids, logits, type_indicator, token_type):

        # Focus on the final 'threshold' tokens
        labels = input_ids[:, -self.threshold:].contiguous()
        logits = logits[:, -self.threshold-1:-1, :].contiguous()
        
        """Compute the loss for the given logits and labels."""
        criterion_mode = getattr(self.criterion, 'mode', None)
        if criterion_mode == 'input':
            type_indicator = type_indicator.repeat_interleave(self.threshold)
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1), type_indicator)
        else:
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Extract log probabilities for the second-to-last token
        second_to_last_logits = logits[:, -2, :]
        log_probs = torch.log_softmax(second_to_last_logits, dim=-1)

        # Get log probability for the target token
        log_prob_target = log_probs.gather(1, token_type.unsqueeze(1)).squeeze(1)

        return loss, log_prob_target


    def train(self, data_loader):
        self.model.train()
        total_loss = 0
        total_samples = 0
        total_neg_log_prob = 0
        self.optimizer.zero_grad()

        for i, batch in enumerate(data_loader):
            input_ids, logits, type_indicator, target_token = self.process_batch(batch, 'target_token')
            loss, log_prob_target = self.compute_loss(input_ids, logits, type_indicator, target_token)
            neg_log_prob = -log_prob_target.mean().item()

            total_samples += len(batch)
            total_loss += loss.item() * len(batch)
            total_neg_log_prob += neg_log_prob * len(batch)

            loss = loss / self.gradient_accumulation  # Normalizing loss for accumulation steps
            loss.backward()

            if (i + 1) % self.gradient_accumulation == 0:  # Optimizer step after 'accumulation_steps' batches
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                if self.verbose:
                    print(f"Batch {i+1} Loss: {loss.item() * self.gradient_accumulation}")
                    print(f"Batch {i+1} Negative Log Probability: {neg_log_prob}")

                torch.cuda.empty_cache()

        average_loss = total_loss / total_samples
        average_neg_log_prob = total_neg_log_prob / total_samples

        return average_loss, average_neg_log_prob

    def evaluate(self, data_loader, token_type='target_token', return_decoded_text=False, max_new_tokens=1):
            self.model.eval()
            total_loss = 0
            total_samples = 0
            total_neg_log_prob = 0
            all_log_probs = []
            all_type_indicator = []
            all_target = []
            decoded_texts = []  # List to store decoded texts

            with torch.no_grad():
                for batch in data_loader:
                    input_ids, logits, type_indicator, target_token = self.process_batch(batch, token_type)
                    loss, log_prob_target = self.compute_loss(input_ids, logits, type_indicator, target_token)
                    neg_log_prob = -log_prob_target.mean().item()

                    total_samples += len(batch)
                    total_loss += loss.item() * len(batch)
                    total_neg_log_prob += neg_log_prob * len(batch)

                    # Collect log probabilities for later concatenation
                    all_log_probs.append(log_prob_target.detach().cpu())
                    all_type_indicator.append(type_indicator)
                    all_target.extend(self.tokenizer.batch_decode(batch['target_token'], skip_special_tokens=False))  # Append decoded target_token to all_target

                    if return_decoded_text:
                        # Decode logits to text
                        # Adjust input_ids and attention_mask to match the shape
                        truncated_input_ids = input_ids[:, :-2]
                        truncated_attention_mask = batch['attention_mask'][:, :-2].to(self.device)

                    # Decode logits to text
                        generated_ids = self.model.generate(
                        input_ids=truncated_input_ids.to(self.device),
                        attention_mask=truncated_attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=0.001,  # Ensure temperature is set here
                        do_sample=False  # Use greedy decoding
                    )
                        decoded_batch_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
                        decoded_texts.extend(decoded_batch_texts)

            average_loss = total_loss / total_samples
            average_neg_log_prob = total_neg_log_prob / total_samples
            all_log_probs = torch.cat(all_log_probs)
            all_type_indicator = torch.cat(all_type_indicator)

            if return_decoded_text:
                return average_loss, average_neg_log_prob, all_log_probs, all_type_indicator, all_target, decoded_texts
            else:
                return average_loss, average_neg_log_prob, all_log_probs, all_type_indicator, all_target
    
    # def batch_generate_text(self, batch):
    #     """Process a single batch of data, focusing only on generating text from the final token."""
    #     input_ids = batch['input_ids'].to(self.device)
    #     attention_mask = batch['attention_mask'].to(self.device)

    #     # Get logits from the model
    #     logits = self.model(input_ids, attention_mask).logits

    #     # Get the last logits for generating text (last token predictions)
    #     last_logits = logits[:, -self.threshold:, :]

    #     # Convert logits to predicted token IDs using argmax
    #     predicted_token_ids = torch.argmax(last_logits, dim=-1)

    #     # Decode the predicted token IDs to text
    #     decoded_texts = self.tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)

    #     # You can return the decoded texts directly or further process them as needed
    #     return decoded_texts, batch['type_indicator']
    
    # def batch_generate_text(self, batch):
    #     """Process a single batch of data, focusing only on generating text from the final token."""
    #     input_ids = batch['input_ids'].to(self.device)
    #     attention_mask = batch['attention_mask'].to(self.device)

    #     # Get logits from the model
    #     logits = self.model(input_ids, attention_mask).logits

    #     # Get the last logits for generating text (last token predictions)
    #     last_logits = logits[:, self.threshold, :]

    #     # Convert logits to predicted token IDs using argmax
    #     predicted_token_ids = torch.argmax(last_logits, dim=-1)

    #     # Decode the predicted token IDs to text
    #     decoded_texts = self.tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)

    #     # You can return the decoded texts directly or further process them as needed
    #     return decoded_texts, batch['type_indicator']
    
    # def compute_recall(self, data_loader):
    #     self.model.eval()
    #     all_yes_status = []
    #     all_type_indicators = []

    #     with torch.no_grad():
    #         for batch in data_loader:
    #             sentences, type_indicators = self.batch_generate_text(batch)
    #             batch_yes_status = ["Yes" in sentence for sentence in sentences]
    #             all_yes_status.extend(batch_yes_status)
    #             all_type_indicators.extend(type_indicators.cpu().numpy())  # Assuming type_indicator is a tensor

    #     # Calculate metrics for each type indicator
    #     type_0_no_fraction = self.calculate_fraction(all_yes_status, all_type_indicators, target_type=0, search_for="No")
    #     type_1_yes_fraction = self.calculate_fraction(all_yes_status, all_type_indicators, target_type=1, search_for="Yes")

    #     return [type_0_no_fraction, type_1_yes_fraction]
    
    # @staticmethod
    # def calculate_fraction(yes_status, type_indicators, target_type, search_for):
    #     indices = [i for i, x in enumerate(type_indicators) if x == target_type]
    #     if indices:
    #         if search_for == "Yes":
    #             relevant_status = [yes_status[i] for i in indices]
    #         else:  # Assume "No"
    #             relevant_status = [not yes_status[i] for i in indices]
    #         fraction = sum(relevant_status) / len(relevant_status)
    #     else:
    #         fraction = None  # No samples of this type_indicator
    #     return fraction



