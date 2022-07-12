import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import dalex as dx

# Read the DataFrame, first using the feature data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add a target column, and fill it with the target data
df['target'] = data.target

# Show the first five rows
df.head()

# Set up the data for modelling
y = df['target'].to_frame()  # define Y
X = df[df.columns.difference(['target'])]  # define X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  # create train and test

# build model - Xgboost
xgb_mod = xgb.XGBClassifier(random_state=42, gpu_id=0)  # build classifier
xgb_mod = xgb_mod.fit(X_train, y_train.values.ravel())

# make prediction and check model accuracy
y_pred = xgb_mod.predict(X_test)

# Utilizing our same xgb_mod model object created above
############## load packages ############


explainer = dx.Explainer(xgb_mod, X, y)  # create explainer from Dalex

############## visualizations #############
# Generate importance plot showing top 30
explainer.model_parts().plot(max_vars=30)

# Generate breakdown plot
explainer.predict_parts(X.iloc[79, :]).plot(max_vars=15)

# Generate SHAP plot
explainer.predict_parts(X.iloc[79, :], type="shap").plot(min_max=[0, 1], max_vars=15)

####### start Arena dashboard #############
# create empty Arena
arena = dx.Arena()

# push created explainer
arena.push_model(explainer)

# push whole test dataset (including target column)
arena.push_observations(X_test)

# run server on port 9294
arena.run_server(port=9291)
