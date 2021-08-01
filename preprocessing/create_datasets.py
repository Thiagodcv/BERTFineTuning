"""
The create_datasets module. Defines subclasses of torch.utils.data.Dataset particular to datasets.
For example, a CookieTheftDataset is defined for DementiaBank transcripts in which subjects participate 
in the picture (Cookie Theft) task
"""
from torch.utils.data import Dataset
import torch

class CookieTheftDataSet(Dataset):
  """A dataset defined for CANARY transcripts in which subjects participate in the Cookie-Theft task. """

  def __init__(self, corpus, labels):
    """
    Parameters
    ----------
    corpus : list of strings
      Each string is a transcript from a given patient
    labels : list of integers: 
      An integer is 1 if a given subject is AD-positive, 0 otherwise
    """
    self.corpus = corpus
    self.labels = labels

  def __len__(self):
    """
    Retrieves the length of the data contained in the dataset.

    Returns
    -------
    int
    """
    return len(self.corpus)

  def __getitem__(self, idx):
    """
    Returns a subset of self.corpus and a subset of self.labels corresponding
    to the requested indices idx.

    Parameters
    ----------
    idx: list of int OR torch.tensor containing only int
      A list of indices

    Returns
    -------
    list
    list
    """
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    sample = self.corpus[idx]
    lbls = self.labels[idx] 

    return sample, lbls