from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from post_processing import prepare_ml_data_set

if __name__ == '__main__':
    XY = prepare_ml_data_set(keep_draws=False)

    X = XY[XY.columns.difference(['outcome'])]
    Y = XY['outcome']
    Y=Y.astype('int')

    reg = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000).fit(X, Y)
