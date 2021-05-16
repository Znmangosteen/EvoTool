class random_forest_model(object):

    def __init__(self, ):
        self.params = {

            'n_estimators': 50,  # 树的数量
            'max_leaf_nodes': range(50, 70, 10),
            'max_depth': range(10, 13, 1),
            'ccp_alpha': 0.0,
            'max_features': 'log2',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.,
            'random_state': 233,
            'feature_num': '',
        }
