import torch
import numpy as np 

def predict(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode.
    predictions = []
    targets = []
    val_indicators = []  # List to collect val_indicator if it exists

    with torch.no_grad():  # No gradients needed for predictions.
        for batch in data_loader:
            # Extract labels
            labels = batch['labels']
            # Extract inputs and move them to the specified device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Perform prediction
            outputs = model(input_ids, attention_mask).logits
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]  # Get probabilities for the second class
            probs_float32 = probs.to(torch.float32)  # Ensure precision

            # Collect outputs
            predictions.extend(probs_float32.cpu().numpy())
            targets.extend(labels)

            # Check if 'val_indicator' exists in the batch and collect it
            if 'val_indicator' in batch:
                val_indicators.extend(batch['val_indicator'])

    # Convert lists to numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Return val_indicators only if they were collected
    if val_indicators:
        val_indicators = np.array(val_indicators)
        return predictions, targets, val_indicators
    else:
        return predictions, targets


def log_predictions(model, tokenizer, device, epoch, dataset, file_name):
    # Open the file for appending text
    with open(file_name, 'a') as f:
        # Iterate through each message in the dataset
        f.write(f"Epoch: {epoch}" + '\n')
        for i in dataset["messages"]:
            x = i[:2]
            # Prepare the inputs for the model
            inputs =  tokenizer.apply_chat_template(x, add_generation_prompt=True, return_tensors='pt').to(device)
            # Generate outputs from the model
            outputs =  model.generate(inputs, max_new_tokens=25)
            # Decode the generated outputs to text
            text = tokenizer.batch_decode(outputs)[0].split('<|assistant|>')[1]
            # Ignore new lines
            text = text.replace('\n', '')
            # Write the generated text to the file
            f.write(text + '\n')

def calculate_fraction(self, yes_status, type_indicators, target_type, search_for):
    indices = [i for i, x in enumerate(type_indicators) if x == target_type]
    if indices:
        if search_for == "Yes":
            relevant_status = [yes_status[i] for i in indices]
        else:  # Assume "No"
            relevant_status = [not yes_status[i] for i in indices]
        fraction = sum(relevant_status) / len(relevant_status)
    else:
        fraction = None  # No samples of this type_indicator
    return fraction

        