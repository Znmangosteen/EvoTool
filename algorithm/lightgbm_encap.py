from lightgbm import LGBMRegressor
from prediction_model import prediction_model


class lightgbm_model(prediction_model):

    def __init__(self, params):
        super(lightgbm_model, self).__init__()
        self.params = params
        self.model = LGBMRegressor(**params)
        self.model.fit()
