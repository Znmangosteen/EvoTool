import numpy as np
from sklearn import preprocessing

class lssvm_model:
    def __init__(self, svm_data, sigma, gamma):
        x_train, y_train = svm_data
        # x_train = pd.DataFrame(self.normalize_data([x_train])[0])
        # y_train = pd.DataFrame(self.normalize_data([y_train])[0])
        x_train_norm, y_train_norm = np.mat(x_train), np.mat(y_train).T
        # x_test_norm, y_test_norm = np.mat(x_test), np.mat(y_test).T
        self.x_ref = x_train_norm
        self.sigma = sigma

        self.alphas, self.b = self.train_lssvm_model(x_train_norm, y_train_norm, gamma, sigma)

    def train_lssvm_model(self, x_train, y_train, gamma, sigma):
        length = np.shape(x_train)[0]
        pattern = self.calculate_similarity(x_train, x_train, sigma)

        # Add gamma to generalize prediction and prevent overfitting issues
        innerMatrix = pattern + np.identity(length) * (1 / gamma)

        # Prepare matrix parts
        leftOnes = np.mat(np.ones((length, 1)))  # [[1][1][1][etc]]
        zeroEntry = np.mat(np.zeros((1, 1)))  # [[0]]
        topOnes = leftOnes.T  # [[1 1 1 etc]]

        # Create final matrices
        topPart = np.hstack((zeroEntry, topOnes))
        botPart = np.hstack((leftOnes, innerMatrix))
        matrix = np.vstack((topPart, botPart))
        solution = np.vstack((zeroEntry, y_train))

        # Calculate bias and alpha values
        b_alpha = matrix.I * solution  # Inverse matrix imprinted on solution vector form
        b = b_alpha[0, 0]
        alphas = b_alpha[1:, 0]

        return alphas, b

    def calculate_similarity(self, x_ref, x_check, sigma):
        # Compares new and reference data, for all combinations of data rows
        rows, columns = np.shape(x_ref)[0], np.shape(x_check)[0]
        pattern = np.mat(np.zeros((rows, columns)))
        for column, item in enumerate(x_check):
            for row, reference_item in enumerate(x_ref):
                pattern[row, column] = self.kernel(v1=item, v2=reference_item)
        return pattern

    def kernel(self, v1, v2):
        # Kernel -> Quantifies how equal one (input) row of data is to another (input) row
        deltaRow = self.squeeze_data(v1 - v2)
        temp = np.dot(deltaRow, deltaRow)
        kernel = np.exp(temp / (-1 * self.sigma ** 2))
        return kernel

    def predict(self, x_check):

        # x_check = pd.DataFrame(self.normalize_data([x_check])[0])
        # Predicts unknown values using known model values (=alphas and bias)
        pattern = self.calculate_similarity(self.x_ref, np.mat(x_check), self.sigma)
        columns = np.shape(pattern)[1]
        prediction = np.mat(np.zeros((columns, 1)))
        for row_index, column_index in enumerate(range(columns)):
            pattern_column = self.squeeze_data(pattern[:, column_index])
            prediction[row_index] = np.dot(pattern_column, self.alphas) + self.b
        return list(np.array(prediction).reshape(-1))

    def squeeze_data(self, data):
        # Convert from matrix to array form
        data_array = np.squeeze(np.asarray(data))
        return data_array

    def normalize_data(self, data):
        # Normalize data based on its min and max values
        scalar = preprocessing.MinMaxScaler()
        result = []
        for item in data:
            norm = scalar.fit_transform(item)
            result.append(norm)
        return result

    def lookup_real_value(self, xmin, xmax, x):
        # Find equivalent real values from normalized values
        result = []
        for norm in x:
            real = xmin + norm * (xmax - xmin)
            result.append(real)
        return result
