import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

def main():
    #load penguin dataset from seaborn library
    penguin_data = sns.load_dataset('penguins')

    print(penguin_data.head())

    #make random forest classifier for penguin dataset
    rf = RandomForestClassifier()
    #fit the classifier to the data
    rf.fit(penguin_data.drop('species', axis=1), penguin_data['species'])

    #make predictions on the penguin dataset
    predictions = rf.predict(penguin_data.drop('species', axis=1))
    print(predictions)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    exit(0)