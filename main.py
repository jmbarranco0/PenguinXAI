import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def main():
    #load penguin dataset from seaborn library
    penguin_data = sns.load_dataset('penguins')

    print(penguin_data)

    species_dic = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
    island_dic = {'Torgersen': 0, 'Biscoe': 1, 'Dream': 2}
    gender_dic = {'Female': 0, 'Male': 1}

    penguin_data['species'] = penguin_data['species'].replace(species_dic)
    penguin_data['island'] = penguin_data['island'].replace(island_dic)
    penguin_data['sex'] = penguin_data['sex'].replace(gender_dic)

    penguin_data = penguin_data.dropna()

    print(penguin_data)

    # Putting feature variable to X
    x = penguin_data.drop('species', axis=1)
    y = penguin_data['species']

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)
    print(x_train.shape, x_test.shape)

    #make random forest classifier for penguin dataset
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    #fit the classifier to the data
    rf.fit(x_train, y_train)

    #make predictions on the penguin dataset
    #predictions = rf.predict(penguin_data.drop('species', axis=1))
    #show random forest accuracy
    print('Random Forest Accuracy:', rf.score(penguin_data.drop('species', axis=1), penguin_data['species']))
    #check if random forest classifier is not overfitting
    print('Random Forest Overfitting:', 1 - rf.score(penguin_data.drop('species', axis=1), penguin_data['species']))

    from sklearn.tree import plot_tree
    plt.figure(figsize=(10, 5))
    plot_tree(rf.estimators_[0], feature_names=x.columns, class_names=['Adelie', 'Chinstrap', 'Gentoo'], filled=True)
    plt.show()

    # Importing the module for LimeTabularExplainer
    import lime.lime_tabular

    # Instantiating the explainer object by passing in the training set, and the extracted features
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(x_train,
                                                            feature_names=x,
                                                            verbose=True, mode='classification')
    # Index corresponding to the test vector
    i = 10

    # Number denoting the top features
    k = 6

    # Calling the explain_instance method by passing in the:
    #    1) ith test vector
    #    2) prediction function used by our prediction model('reg' in this case)
    #    3) the top features which we want to see, denoted by k
    exp_lime = explainer_lime.explain_instance(
        x_test[i], rf.predict, num_features=k)

    # Finally visualizing the explanations
    exp_lime.show_in_notebook()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    exit(0)