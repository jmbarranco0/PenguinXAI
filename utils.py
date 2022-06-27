import pickle


def save_model(model, filename):
    with open("models/" + filename, 'wb') as f:
        pickle.dump(model, f)
    print('Model saved to {}'.format(filename))


def load_model(filename):
    with open("models/" + filename, 'rb') as f:
        model = pickle.load(f)
    return model
