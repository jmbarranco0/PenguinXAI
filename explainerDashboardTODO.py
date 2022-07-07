from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard  # TODO: importar ClassifierExplainer y ExplainerDashboard de la libreria explainerdashboard
from explainerdashboard.datasets import titanic_survive, titanic_names

feature_descriptions = {
    "Sex": "Sexo del pasajero",
    # TODO: Añadimos una descripción a la variable "Gender"
    "Deck": "La cubierta en la que el/la pasajero/a tenía su cabina.",
    "PassengerClass": "La clase del ticket: 1a, 2a or 3a clase",
    # TODO: Añadimos una descripción a la variable "Fare" (cantidad de dinero pagado)
    "Embarked": "El puerto en el que el pasajero se montó en el Titanic. Puede ser Southampton, Cherbourg o Queenstown",
    # TODO: Añadimos una descripción a la variable "Age"
    "No_of_siblings_plus_spouses_on_board": "La suma del número de hermanos más el número de cónyuges a bordo",
    "No_of_parents_plus_children_on_board": "La suma del número de padres más el número de niños a bordo",
}

X_train, y_train, X_test, y_test = None  # TODO: Cargamos el dataset de titanic que nos proporciona explainerDashboard
train_names, test_names = titanic_names()

model = RandomForestClassifier(n_estimators=50, max_depth=5)  # TODO: Creamos el modelo de Random Forest con 50 árboles y una profundidad de 5
# TODO: Entrenamos el modelo con los datos de entrenamiento

explainer = ClassifierExplainer(model, X_test, y_test,
                                cats=['Deck', 'Embarked',
                                      {'Gender': ['Sex_male', 'Sex_female', 'Sex_nan']}],
                                cats_notencoded={'Embarked': 'Stowaway'},
                                # TODO: Indicamos las descripciones de las características (var: descripcion)
                                # TODO: Indicamos las etiquetas de los datos (var: label)
                                # TODO: Indicamos los identificadores que tendrán las instancias de los datos de entrenamiento (var: idxs)
                                # TODO: Indicamos el nombre por defecto que tendrán las instancias (var: index_name)
                                # TODO: Indicamos el nombre de la variable objetivo (var: target)
                                )

db = ExplainerDashboard(explainer,
                        # TODO: Indicamos un título para el dashboard (var: title)
                        )
# TODO: Lanzamos el dashboard en el puerto 8050

#  References:
#  https://explainerdashboard.readthedocs.io/en/latest/
#  https://github.com/oegedijk/explainerdashboard
