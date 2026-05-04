from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_transformation import calculate_labels_alarm
from utils import cusum_detection, detect_change_point
from models.autoencoder import AutoencoderModel, Autoencoder

# https://klaviyo.tech/developing-our-first-anomaly-detection-algorithm-7c84cab7ca46
# https://blog.stackademic.com/the-cusum-algorithm-all-the-essential-information-you-need-with-python-examples-f6a5651bf2e5
class AutoencoderAlarmModel(AutoencoderModel):
    """ Class for Autoencoder with alarm model"""
    
    def _calculate_labels(self, df, contaminant, window_size):
        return calculate_labels_alarm(df, contaminant, window_size)
    
    def _post_predictions(self, y_pred):
        return detect_change_point(y_pred, count_required=20)
    