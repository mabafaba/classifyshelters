from keras import backend as K


smooth = 1.0

# # dice coefficient
def dice_coef(truth, prediction):
    n_true_positive  = sum(prediction[truth == 1])
    n_false_positive = sum(prediction[truth == 0])
    n_false_negative = sum(1-prediction[truth == 1])
    dice = 2 * n_true_positive / (2*n_true_positive + n_false_positive + n_false_negative)
    return(dice)


# loss is negative dice coefficient
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


