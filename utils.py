import pickle
import pandas as pd


def save_model(model, filename):
    with open("models/" + filename, 'wb') as f:
        pickle.dump(model, f)
    print('Model saved to {}'.format(filename))


def load_model(filename):
    with open("models/" + filename, 'rb') as f:
        model = pickle.load(f)
    return model


def save_x_train_and_x_test(x_train, x_test, filename):
    x_train.to_csv("data/" + filename + "_x_train.csv")
    x_test.to_csv("data/" + filename + "_x_test.csv")
    print('Data saved to {}'.format(filename))


def load_x_train_and_x_test(filename):
    x_train = pd.read_csv("data/" + filename + "_x_train.csv")
    x_test = pd.read_csv("data/" + filename + "_x_test.csv")
    return x_train, x_test
