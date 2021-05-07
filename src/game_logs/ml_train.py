from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from post_processing import prepare_ml_data_set

if __name__ == '__main__':
    XY = prepare_ml_data_set(keep_draws=False)

    X = XY[XY.columns.difference(['outcome'])]
    Y = XY['outcome']
    Y=Y.astype('int')

    # mlp = MLPRegressor(solver='sgd', max_iter=100, activation='relu',
    #                     random_state=1, learning_rate_init=0.01,
    #                     batch_size=X.shape[0], hidden_layer_sizes = (50,))
    # mlp.fit(X, Y)

    reg = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000).fit(X, Y)
