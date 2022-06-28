import seaborn as sns
import shap
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from utils import save_model, save_x_train_and_x_test


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
    save_x_train_and_x_test(x_test, y_test, 'penguin')

    print(x_train.shape, x_test.shape)

    #make random forest classifier for penguin dataset
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    #fit the classifier to the data
    rf.fit(x_train, y_train)

    rf_preds = rf.predict(x_test)
    print('The accuracy of the Random Forests model is :\t', metrics.accuracy_score(rf_preds, y_test))

    from sklearn.tree import plot_tree
    plt.figure(figsize=(10, 5))
    plot_tree(rf.estimators_[0], feature_names=x.columns, class_names=['Adelie', 'Chinstrap', 'Gentoo'], filled=True)
    plt.show()

    save_model(rf, 'penguin_rf_model.pkl')

    shap.initjs()
    explainer = shap.TreeExplainer(rf)

    single_shap_value = explainer(x_test.sample(n=1))
    shap.summary_plot(single_shap_value, feature_names=x_test.columns, plot_type='bar')
    plt.show()

    # Visualize all values
    shap_values = explainer.shap_values(x_train)
    shap.summary_plot(shap_values, x_train, feature_names=x_train.columns, plot_type='bar')
    plt.show()

    shap.force_plot(explainer.expected_value[0], shap_values[0:5, :], x_test.iloc[0:5, :], plot_cmap="DrDb",
                    feature_names=x.columns)


if __name__ == '__main__':
    main()
    exit(0)