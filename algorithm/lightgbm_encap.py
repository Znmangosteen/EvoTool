from lightgbm import LGBMRegressor

from mertric import r_square
from prediction_model import prediction_model


class lightgbm_model(prediction_model):

    def __init__(self, params):
        super(lightgbm_model, self).__init__()
        self.params = params
        self.model = LGBMRegressor(**params)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_val, y_val)], eval_metric=['rmse', r_square])
        evals_result = self.model.evals_result_
        return evals_result['training']['rmse'], evals_result['training']['r_square'], evals_result['valid_1']['rmse'], \
               evals_result['valid_1']['r_square']

    def get_feature_importance(self,):
        return self.model.feature_importances_
