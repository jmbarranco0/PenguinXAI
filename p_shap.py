import shap
import seaborn as sns
from utils import load_model, load_x_train_and_x_test
import IPython  # Is needed to do shap.initjs() before using shap.summary_plot()


def main():
    rf = load_model('penguin_rf_model.pkl')
    x_train, x_test = load_x_train_and_x_test('penguin')

    penguin_data = sns.load_dataset('penguins')
    # Get feature names from penguin data and store it in feature_names
    feature_names = penguin_data.drop('species', axis=1).columns

    shap.initjs()
    explainer = shap.TreeExplainer(rf)

    instance_index = 84

    print(penguin_data.loc[[instance_index]])

    chosen_instance = x_test.loc[[instance_index]]
    shap_values = explainer.shap_values(chosen_instance)
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values[1], chosen_instance)


if __name__ == '__main__':
    main()
    exit(0)