"""
The evaluate module. Used for evaluating models with performance metrics.
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import settings

LOAD_PATH = settings.LOAD_PATH
CHECKPOINT = settings.CHECKPOINT

def evaluate_scores(corpus, labels, load_path=LOAD_PATH):
    """
    Evaluates pre-trained model accuracy, f1, auc, precision and recall on a given dataset. Due to
    GPU memory constraints, try to use evaluation datasets of n < 200. 

    Parameters
    ----------
    corpus: list of str
    labels: list of [0,1]
    load_path: str
        Path to load model architecture & weights from

    Returns
    -------
    dict

    """
    with torch.no_grad():
        model = AutoModelForSequenceClassification.from_pretrained(load_path)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print('cuda available?: ', torch.cuda.is_available())
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        full_batch = tokenizer(corpus, padding=True, truncation=True, return_tensors="pt")
        full_batch = {k: v.to(device) for k,v in full_batch.items()}
        output = model(**full_batch)
        preds = torch.argmax(output['logits'], dim=1)

        acc = accuracy_score(labels, preds.tolist())
        f1 = f1_score(labels, preds.tolist())
        precision = precision_score(labels, preds.tolist())
        recall = recall_score(labels, preds.tolist())
        roc_auc = roc_auc_score(labels, preds.tolist())

    print("Accuracy: {}, F1: {}, Precision: {}, Recall: {}, AUC: {}".format(acc, f1, precision, recall, roc_auc))

    return {'Accuracy': acc, 'F1': f1, 'Precision': precision, 'Recall': recall, 'AUC': roc_auc}