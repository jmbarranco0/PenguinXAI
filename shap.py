import seaborn as sns
import shap
import xgboost as xgboost
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import openpyxl  # Needed for load dataset

from utils import load_dataset_from_xlsx, save_model, save_x_train_and_x_test


def main():
    #load pumpkin dataset from /data
    pumpkin_data = load_dataset_from_xlsx('data/Pumpkin_Seeds_Dataset.xlsx')

    feature_names = pumpkin_data.drop('Class', axis=1).columns
    label_names = pumpkin_data['Class'].unique()

    # Replace Class label with numeric value
    label_dic = {'Çerçevelik': 0, 'Ürgüp Sivrisi': 1}
    pumpkin_data['Class'] = pumpkin_data['Class'].replace(label_dic)

    # Put feature variable to X
    x = pumpkin_data.drop('Class', axis=1)
    y = pumpkin_data['Class']

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

    # Make random forest classifier for pumpkin dataset
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the classifier to the data
    rf.fit(x_train, y_train)

    rf_preds = rf.predict(x_test)

    print('The accuracy of the Random Forests model is :\t', metrics.accuracy_score(rf_preds, y_test))

    from sklearn.tree import plot_tree
    plt.figure(figsize=(100, 50))
    plot_tree(rf.estimators_[0], feature_names=x.columns, class_names=['Adelie', 'Chinstrap', 'Gentoo'], filled=True)
    plt.show()

    xgtree = xgboost.DMatrix(x, label=y)
    model = xgboost.train({
        'eta': 1, 'max_depth': 3, 'base_score': 0, "lambda": 0
    }, xgtree, 1)
    print("Model error =", np.linalg.norm(y - model.predict(xgtree)))
    print(model.get_dump(with_stats=True)[0])

    predictions = model.predict(xgtree, output_margin=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(xgtree)

    shap.summary_plot(shap_values, x)

    shap.dependence_plot("Area", shap_values, x)

    shap.initjs()
    shap.dependence_plot("Aspect_Ration", shap_values, x)  # Aspect_Ratio is determining

    shap.dependence_plot("Solidity", shap_values, x)  # Height is determining

    # Visualize all values
    shap_values = explainer.shap_values(x_train)
    shap.summary_plot(shap_values, x_train, feature_names=x_train.columns, plot_type='bar')
    plt.show()

    xgboost.plot_importance(model)
    plt.show()

    shap_values_ind = shap.TreeExplainer(model).shap_values(x)
    for name in x_train.columns:
        shap.dependence_plot(name, shap_values_ind, x, display_features=X_display)

    save_model(rf, 'pumpkin_rf_model.pkl')


if __name__ == '__main__':
    main()
    exit(0)