class prediction_model(object):
    def __init__(self, ):
        self.feature_importance = []
        self.model = None

    def predict(self, X):
        return self.model.predict(X)
