############################################# fine_tune_bert.py settings #############################################

"""
Random state settings
---------------------
WEIGHT_INIT_SEED: int
    Random seed used for initializing the top linear layer in BertForSequenceClassification
DATA_ORDER_SEED: int
    Random seed for order in which data is sampled from the dataloader. 
"""
WEIGHT_INIT_SEED = 2
DATA_ORDER_SEED = 2

"""
Training settings
-----------------
BATCH_SIZE: int
    Batch size for training
LEARNING_RATE: float
    The learning rate
NUM_EPOCHS: int
    Number of epochs to train the model for
NUM_WARMUP_STEPS: int
    Number of epochs for warmup during training

"""
BATCH_SIZE = 8
LEARNING_RATE = 7e-7
NUM_EPOCHS = 150
NUM_WARMUP_STEPS = 0

"""
Saving & loading settings
-------------------------
OLD_EPOCH: int
    Epoch number to begin training from
PATIENCE: int
    Number of epochs without significant improvement in validation score before training terminates
SAVE_MODELS_PATH: str
    Path to save model to
SAVE_RESULTS_PATH: str
    Path to save training logs to
LOAD_PATH: str
    Path to load models from
CHECKPOINT: str
    HuggingFace checkpoint to load models and other preprocessing tools
RESULTS_FILE_NAME: str
    Name of results file to be saved
"""
OLD_EPOCH = 0
PATIENCE = 10
SAVE_MODELS_PATH = '/home/thiago/BERTFineTuning/saves/models'
SAVE_RESULTS_PATH = '/home/thiago/BERTFineTuning/saves/training_log'
LOAD_PATH = None
CHECKPOINT = 'bert-base-cased'
RESULTS_FILE_NAME = 'results.txt'

"""
Regularization settings
-----------------------
HIDDEN_DROPOUT_PROB: float in [0, 1]
    Dropout probability for hidden layers
ATTENTION_DROPOUT_PROB: float in [0, 1]
    Dropout probability for attention heads
"""
HIDDEN_DROPOUT_PROB = 0.1
ATTENTION_DROPOUT_PROB = 0.1
