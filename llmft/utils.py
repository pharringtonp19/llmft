import torch
import numpy as np 

def predict(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode.
    predictions = []
    targets = []
    with torch.no_grad():  # No gradients needed for predictions.
        for batch in data_loader:
            labels = batch['labels']
            # Assuming your batch only includes input data and not labels.
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask).logits  # Get model outputs
            probs = torch.nn.functional.softmax(outputs, dim=1)[:,1]
            probs_float32 = probs.to(torch.float32)  # Convert to float32 before calling .numpy()
            predictions.extend(probs_float32.cpu().numpy())
            targets.extend(labels)

    return np.array(predictions), np.array(targets)


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

        