import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def evaluate(logits, targets):
    """
    Đánh giá model sử dụng accuracy và f1 scores.
    Args:
        logits (B,C): torch.LongTensor. giá trị predicted logit cho class output.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        acc (float): the accuracy score
        f1 (float): the f1 score
    """
    # Tính accuracy score và f1_score
    logits = logits.detach().cpu().numpy()    
    y_pred = np.argmax(logits, axis = 1)
    targets = targets.detach().cpu().numpy()
    f1 = f1_score(targets, y_pred, average='weighted')
    acc = accuracy_score(targets, y_pred)
    return acc, f1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logits = torch.tensor([[0.1, 0.2, 0.7],
                       [0.4, 0.1, 0.5],
                       [0.1, 0.2, 0.7]]).to(device)
targets = torch.tensor([1, 2, 2]).to(device)
evaluate(logits, targets)