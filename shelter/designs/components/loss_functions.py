from keras import backend as K 
# import numpy as np


# dice coefficient: value is between 0 and 1 
# (see: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
# Closer to 0 means two distributions are very dissimilar, 1 the distributions are perfectly identical.
# returns ~0 for all wrong
# returns ~0 for prediction all 0
# returns 1 for all correct
# penalizes false positives
# The smooth is inserted to avoid having value of exactly 0, and they are inserted in both in denom and num so max value is still 1.
smooth = 1.0
def dice_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

# loss is to be minimized so we take the negative of the dice coefficient
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
