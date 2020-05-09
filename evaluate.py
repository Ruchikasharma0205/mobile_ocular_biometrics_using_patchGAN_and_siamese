import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import os
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

prediction = list(np.array(pd.read_csv('prediction.csv', header=None))[:, 0])
actual = list(np.array(pd.read_csv('actual.csv', header=None))[:, 0])

m = tf.keras.metrics.SensitivityAtSpecificity(0.99999)
m.update_state(actual, prediction)
print('Final result: ', m.result().numpy())
# print(actual, prediction)
fpr, tpr, thresholds = roc_curve(actual, prediction)

# print(fpr)
# print(tpr)
# print(thresholds)

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

plot_roc_curve(fpr, tpr)