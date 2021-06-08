import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, log_loss, \
    confusion_matrix


def eval_model(model, data_x, data_y, ):
    model_pred = model.predict(data_x)
    return np.square(mean_squared_error(data_y, model_pred)), r2_score(data_y, model_pred)


def eval_model_classification(model, data_x, data_y, ):
    model_pred = model.predict(data_x)
    return accuracy_score(data_y, model_pred), confusion_matrix(data_y, model_pred)


def r_square(y_true, y_pred):
    return 'r_square', r2_score(y_true, y_pred), True
