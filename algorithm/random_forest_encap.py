from sklearn.ensemble import RandomForestRegressor

from mertric import eval_model
from prediction_model import prediction_model


class random_forest_model(prediction_model):

    def __init__(self, params):
        super(random_forest_model, self).__init__()
        self.params = params
        self.model = RandomForestRegressor(**params)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        rmse_train = []
        rmse_val = []
        r2_train = []
        r2_val = []

        # rfr_eval = copy.deepcopy(rfr)
        es_bk = self.model.estimators_
        for i in range(len(self.model.estimators_)):
            rmse, r2 = eval_model(self, X_train, y_train)
            rmse_train = [rmse] + rmse_train
            r2_train = [r2] + r2_train

            rmse, r2 = eval_model(self, X_val, y_val)
            rmse_val = [rmse] + rmse_val
            r2_val = [r2] + r2_val

            self.model.estimators_ = self.model.estimators_[:-1]
        self.model.estimators_ = es_bk
        return rmse_train, r2_train, rmse_val, r2_val




    def get_feature_importance(self):
        return self.model.feature_importances_
