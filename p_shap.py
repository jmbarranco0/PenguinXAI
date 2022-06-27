import dalex as dl
from utils import save_model, load_x_train_and_x_test


def main():
    model = load_model('penguin_rf_model.pkl')
    x_train, x_test = load_x_train_and_x_test('penguin')


if __name__ == '__main__':
    main()
    exit(0)