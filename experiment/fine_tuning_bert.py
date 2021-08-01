"""
The fine_tuning_bert module. Used for fine-tuning a pre-trained BERT model.
"""
import os
import settings
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, get_scheduler, BertConfig, BertForSequenceClassification, BertTokenizer
import torch
from sklearn.metrics import accuracy_score
import settings

LR = settings.LEARNING_RATE
NUM_EPOCHS = settings.NUM_EPOCHS
NUM_WARMUP_STEPS = settings.NUM_WARMUP_STEPS
OLD_EPOCH = settings.OLD_EPOCH
PATIENCE = settings.PATIENCE
WEIGHT_INIT_SEED = settings.WEIGHT_INIT_SEED
DATA_ORDER_SEED = settings.DATA_ORDER_SEED
HIDDEN_DROPOUT_PROB = settings.HIDDEN_DROPOUT_PROB
ATTENTION_DROPOUT_PROB = settings.ATTENTION_DROPOUT_PROB
SAVE_MODELS_PATH = settings.SAVE_MODELS_PATH
SAVE_RESULTS_PATH = settings.SAVE_RESULTS_PATH
LOAD_PATH = settings.LOAD_PATH
CHECKPOINT = settings.CHECKPOINT
RESULTS_FILE_NAME = settings.RESULTS_FILE_NAME

def train_bert(
    train_corpus,
    train_labels,
    val_corpus,
    val_labels,
    dataloader,
    lr = LR, 
    num_epochs = NUM_EPOCHS, 
    num_warmup_steps = NUM_WARMUP_STEPS, 
    old_epoch = OLD_EPOCH, 
    patience = PATIENCE, 
    weight_init_seed = WEIGHT_INIT_SEED, 
    data_order_seed = DATA_ORDER_SEED, 
    hidden_dropout_prob = HIDDEN_DROPOUT_PROB, 
    attention_dropout_prob = ATTENTION_DROPOUT_PROB, 
    save_model_at_end = False, 
    save_models_path = SAVE_MODELS_PATH,  
    save_results_path = SAVE_RESULTS_PATH, 
    load_path = LOAD_PATH, 
    checkpoint = CHECKPOINT, 
    results_file_name = RESULTS_FILE_NAME 
    ):
    """
    Fine-tunes a pre-trained BERT model with a linear layer on top of pooled output.
    Documentation for arguments with default values can be found in settings.py.

    Parameters
    ----------
    train_corpus: list of strings
    train_labels: list of [0,1]
    val_corpus: list of strings
    val_labels: list of [0,1]
    dataloader: torch.util.data.Dataloader

    Returns
    -------
    float
        The highest validation score achieved during a run
    int
        The number of epochs the model was trained for
    """

    """ Loading the model and optimizer """
    if load_path==None:
        torch.manual_seed(weight_init_seed)
        configuration = BertConfig()
        configuration.hidden_dropout_prob = hidden_dropout_prob
        configuration.attention_probs_dropout_prob = attention_dropout_prob
        model = BertForSequenceClassification(configuration)
        optimizer = AdamW(model.parameters(), lr=lr)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(load_path)
        optimizer = AdamW(model.parameters(), lr=lr)

    """ Initializing the variables  """
    num_training_steps = num_epochs * len(dataloader)
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, threshold=0.5)

    """ Enabling GPU services """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('cuda available?: ', torch.cuda.is_available())
    model.to(device)

    """ Fine-tune BERT model """
    torch.manual_seed(data_order_seed)
    top_val = 0
    epochs_since_last_increase = 0
    for epoch in range(old_epoch, num_epochs + old_epoch):
        model.train() 
        total_loss = 0
        for i, (input, labels) in enumerate(dataloader):
            # forwards pass, backward pass, update
            batch = tokenizer(list(input), padding=True, truncation=True, return_tensors="pt") # batch pads dynamically
            batch['labels'] = labels
            try:
                batch = {k: v.to(device) for k,v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            except Exception as e:
                with open(os.path.join(SAVE_RESULTS_PATH, 'error.txt'), 'a') as file:
                    file.write(str(e) + '\n')
                    file.write(str(batch.items()) + '\n')

        lr_scheduler.step(total_loss)

        """ Evaluation on training and validation set after every epoch """
        model.eval() 
        with torch.no_grad():
            train_acc = return_accuracy(train_corpus[:70], train_labels[:70], tokenizer, device, model)
            val_acc = return_accuracy(val_corpus, val_labels, tokenizer, device, model)
            print("Epoch: {}, training accuracy: {}, validation accuracy: {}, total loss: {}".format(epoch, train_acc, val_acc, total_loss)) 

            """ Saving experiment results after every epoch """
            save_results(epoch, train_acc, val_acc, total_loss, results_file_name)

        """ Early stopping mechanism. Increase in validation accuracy must 
            increase by at least 0.01 every ``patience`` number of epochs else 
            terminate training.
        """
        if val_acc > top_val + 0.01: 
            top_val = val_acc
            epochs_since_last_increase = 0
        else:
            epochs_since_last_increase += 1

        if epochs_since_last_increase > patience:
            if save_model_at_end:
                model.save_pretrained(save_models_path)
            return top_val, epoch

    if save_model_at_end:
        model.save_pretrained(save_models_path)

    return top_val, int(epoch)

def return_accuracy(corpus, labels, tokenizer, device, model):
    """
    Computes accuracy of current model on training set/validation set.

    Parameters
    ----------
    corpus: list of str
    labels: list of [0,1]
    data_type: 'train' if training set or 'val' if validation set
    tokenizer: transformer.AutoTokenizer
    epoch: int
    device: torch.device(x) where x = 'gpu' or x = 'cpu' 
    model: transformer.AutoModelForSequenceClassification

    Returns
    -------
    float
        Accuracy on a given set

    """
    full_batch = tokenizer(corpus, padding=True, truncation=True, return_tensors="pt")
    full_labels = torch.tensor(labels).to(device)
    full_batch['labels'] = full_labels
    full_batch = {k: v.to(device) for k,v in full_batch.items()}
    full_output = model(**full_batch)
    preds = torch.argmax(full_output['logits'], dim=1)

    acc = accuracy_score(full_labels.tolist(), preds.tolist())
    return acc

def save_results(epoch, train_results, val_results, loss, results_file_name):
    """
    Saves training and validation accuracy of current model to SAVE_RESULTS_PATH.

    Parameters
    ----------
    epoch: int
    train_results: string
    val_results: string
    loss: float
    results_file_name: string
        The name of the file which stores the results
    """
    if not os.path.exists(os.path.join(SAVE_RESULTS_PATH, results_file_name)):
        with open(os.path.join(SAVE_RESULTS_PATH, results_file_name), 'a') as file:
            file.write('epoch, total loss, train accuracy, validation accuracy' + '\n')

    with open(os.path.join(SAVE_RESULTS_PATH, results_file_name), 'a') as file:
        file.write("{}, {}, {}, {}".format(epoch, loss, train_results, val_results) + '\n')

