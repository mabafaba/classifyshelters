from keras import backend as K


smooth = 1.0

# # dice coefficient
def dice_coef(y_true, y_pred):
    # truth as vector:
    y_true_f = K.flatten(y_true)
    # prediction as vector:
    y_pred_f = K.flatten(y_pred)
    # count predicted "1"s that are also true "1"s = count true positives
    intersection = K.sum(y_true_f * y_pred_f)
    # 2 * count true positives / ( count true "1"s + count predicted "1"s
    # returns 0 for all wrong
    # returns 0 for prediction all 0
    # returns 1 for all correct
    # how 'good' a value in between depends is on total % of true "1"s.
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# loss is negative dice coefficient
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)