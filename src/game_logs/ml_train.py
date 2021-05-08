from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from post_processing import prepare_ml_data_set

if __name__ == '__main__':
    XY = prepare_ml_data_set()

    X = XY[XY.columns.difference(['score'])]
    Y = XY['score']
    Y=Y.astype('int64')

    mlp = MLPRegressor(solver='sgd', max_iter=100000, activation='tanh',
                        random_state=1, learning_rate='adaptive', learning_rate_init=0.1,
                        batch_size=X.shape[0], hidden_layer_sizes = (50, 50,))
    mlp.fit(X, Y)

    # reg = LinearRegression(fit_intercept=False).fit(X, Y)
    # reg.fit(X,Y)

    # print([c for c in reg.coef_])