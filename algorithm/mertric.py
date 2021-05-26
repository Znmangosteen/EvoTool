import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def eval_model(model, data_x, data_y, ):
    model_pred = model.predict(data_x)
    return np.square(mean_squared_error(data_y, model_pred)), r2_score(data_y, model_pred)


def r_square(y_true, y_pred):
    return 'r_square', r2_score(y_true, y_pred), True
