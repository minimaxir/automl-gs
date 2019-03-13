import pandas as pd
import numpy as np
import json
import os
import csv
from datetime import datetime
from math import floor
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score
{% if framework == 'tensorflow' %}
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM, CuDNNLSTM, GRU, CuDNNGRU, concatenate, Dense, BatchNormalization, Dropout, AlphaDropout
from tensorflow.keras import backend as K
from tensorflow.contrib.opt import AdamWOptimizer
from tensorflow.train import cosine_decay
{% endif %}
