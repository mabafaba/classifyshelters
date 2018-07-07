from keras import backend as K 
# import numpy as np

smooth = 1.0

# # dice coefficient
def dice_coef(y_true, y_pred):
    # truth as vector:

    # count predicted "1"s that are also true "1"s = count true positives
    intersection = K.sum(y_true * y_pred)
    # 2 * count true positives / ( count true "1"s + count predicted "1"s
    # returns 0 for all wrong
    # returns 0 for prediction all 0
    # returns 1 for all correct
    # how 'good' a value in between depends is on total % of true "1"s.
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

# loss is negative dice coefficient
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



# loss is negative dice coefficient
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# # dice coefficient
# def dice_coef(truth, prediction):
#     truth = K.flatten(truth)
#     prediction = K.flatten(prediction)
#     truth = np.array(truth)
#     prediction = np.array(prediction)
#     print(prediction)
#     n_true_positive  = sum(prediction[truth == 1])
#     n_false_positive = sum(prediction[truth == 0])
#     n_false_negative = sum(1-prediction[truth == 1])
#     dice = 2 * n_true_positive / (2*n_true_positive + n_false_positive + n_false_negative)
#     tf.convert_to_tensor(dice, np.float32)
#     return(dice)
