import torch

def compute_f1_score(predictions, labels):
    num_classes = len(torch.unique(labels))
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(labels.view(-1), predictions.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    
    precision = torch.diag(confusion_matrix) / confusion_matrix.sum(dim=0).clamp(min=1)
    recall = torch.diag(confusion_matrix) / confusion_matrix.sum(dim=1).clamp(min=1)
    f1_scores = 2 * (precision * recall) / (precision + recall).clamp(min=1)
    weights = confusion_matrix.sum(dim=1) / confusion_matrix.sum()
    weighted_f1 = (weights * f1_scores).sum()
    
    return weighted_f1.item()


def compute_recall(predictions, labels):
    # Calculate the number of classes
    num_classes = len(torch.unique(labels))
    
    # Create confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(labels.view(-1), predictions.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    
    # Calculate recall for each class
    recall = torch.diag(confusion_matrix) / confusion_matrix.sum(dim=1).clamp(min=1)
    
    # Optionally, calculate average recall across all classes
    average_recall = recall.mean().item()  # Average recall for handling imbalanced data better
    
    return recall, average_recall