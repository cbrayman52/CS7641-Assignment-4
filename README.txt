To access the code: go to the following link:
https://github.com/cbrayman52/CS7641-Assignment-2

To run the code, all you have to do is run the submission.py file.

This will perform all experiments and generate all images used in the report.
Images are saved in the 'Images' directory including a subfolder for the specific model being analyzed.
CSVs are saved in the 'Output' directory including a subfolder for the specific model being analyzed.

The Wine Quality Dataset was provided as a csv in the Dataset folder. However it can also be found online here:
https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

Note: Running the code will take 2-4 hours to fully compile depending on the machine. 

Use pip install -r requirements.txt to download the needed libraries.

Libraries used:
import mlrose_hiive
import ast
import time
import string
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mlrose_hiive.algorithms.crossovers import UniformCrossOver
from mlrose_hiive.algorithms.mutators import ChangeOneMutator
from mlrose_hiive.opt_probs.discrete_opt import DiscreteOpt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score