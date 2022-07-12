from alibi.explainers import ALE, plot_ale
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np


def main():
    # load penguin dataset from seaborn library
    penguin_data = sns.load_dataset('penguins')

    print(penguin_data)

    species_dic = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
    island_dic = {'Torgersen': 0, 'Biscoe': 1, 'Dream': 2}
    gender_dic = {'Female': 0, 'Male': 1}

    penguin_data['species'] = penguin_data['species'].replace(species_dic)
    penguin_data['island'] = penguin_data['island'].replace(island_dic)
    penguin_data['sex'] = penguin_data['sex'].replace(gender_dic)

    penguin_data = penguin_data.dropna()

    # Putting feature variable to X
    x = penguin_data.drop('species', axis=1)
    y = penguin_data['species']

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

    print(x_train.shape, x_test.shape)

    # make random forest classifier for penguin dataset
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # fit the classifier to the data
    rf.fit(x_train, y_train)

    rf_preds = rf.predict(x_test)
    print('The accuracy of the Random Forests model is :\t', metrics.accuracy_score(rf_preds, y_test))

    # make ale for penguin dataset
    ale = ALE(rf_preds)

    exp = ale.explain(x_train)
    plot_ale(exp)
    print("Eh")


if __name__ == '__main__':
    main()