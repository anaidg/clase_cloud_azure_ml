import pandas as pd
import numpy  as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pickle

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='.',
        help='Path to the training data'
    )

    args = parser.parse_args()

    df = pd.read_csv(args.data_path, delimiter = ",")
    df=df.drop('Unnamed: 0', axis=1)

    X = df.drop(['id','en_cartera','valor_cartera_mes_ant','valor_cartera'],axis=1) #Estas variables solo son diferentes de cero cuando esta en mora ( en_cartera = 1),
                                                                                  #y al entrenar modelos con ellas el score es perfecto dado que 
                                                                                  #es como entrenar con la variable a predecir, por lo que se toma la decisión sacarlas del dataset
    y = df['en_cartera']

    st = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    Xtest, Xval, ytest, yval = train_test_split(Xtest, ytest, test_size=0.5, random_state=0, stratify=ytest)

    model = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('Classifier',RandomForestClassifier(n_estimators=10, max_depth=6, class_weight='balanced_subsample'))
    ])


    tuned_parameters = [ {'Classifier__n_estimators':[6,8,10], 'Classifier__max_depth':[2,4,6], 'Classifier__max_features':[4,8,12,16]}]
    model_tunning = GridSearchCV(estimator=model, param_grid=tuned_parameters, cv=st, scoring='balanced_accuracy',n_jobs=-1,verbose=2)

    model_tunning.fit(Xtrain, ytrain)

    #predicciones
    y_pred = model_tunning.predict(Xtest)

    #mejor score
    print('Mejor score: '+ str(model_tunning.best_score_))

    #pickle.dump(model_tunning, open('../outputs/gestion_cartera.pkl', 'wb'))
