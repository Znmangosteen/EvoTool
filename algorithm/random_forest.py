from sklearn.ensemble import RandomForestRegressor

from prediction_model import prediction_model


class random_forest_model(prediction_model):

    def __init__(self, params):
        self.model = RandomForestRegressor(**params)

    def train(self, X, y, ):
        self.model.fit(X, y)
        rmse, r2 = [], []

    def eval(self, val_X, val_Y):
        pass

    def get_feature_importance(self):
        return self.model.feature_importances_
