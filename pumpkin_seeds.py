import seaborn as sns
import shap
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from utils import load_dataset_from_xlsx, save_model, save_x_train_and_x_test


def main():
    #load pumpkin dataset from /data
    pumpkin_data = load_dataset_from_xlsx('Pumpkin_Seeds_Dataset.xlsx')

    print(pumpkin_data)

    pumpkin_data = pumpkin_data.dropna()


if __name__ == '__main__':
    main()
    exit(0)