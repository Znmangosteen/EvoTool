class prediction_model(object):
    def __init__(self, ):
        self.feature_importance = []
        self.used_features = []
        self.target = ''
        self.x_scalar = None
        self.y_scalar = None
        self.model = None

    def predict(self, X):
        return self.model.predict(X)
