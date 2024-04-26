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
            predictions.extend(probs.cpu().numpy())
            targets.extend(labels)

    return np.array(predictions), np.array(targets)