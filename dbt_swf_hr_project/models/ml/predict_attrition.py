import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def model(dbt, session):
    dbt.config(
        packages=["pandas", "scikit-learn"]
    )

    # Load the data
    df = dbt.ref("stg_hr_actuals").to_pandas()
    to_pred = dbt.ref("stg_hr_to_predict").to_pandas()
    init_to_pred = to_pred

    # Preparing the training and test data
    df['ATTRITION'] = df['ATTRITION'].apply(lambda x: 1 if x == True else 0)
    df['GENDER'] = df['GENDER'].apply(lambda x: 1 if x == 'Male' else 0)
    df['OVER18'] = df['OVER18'].apply(lambda x: 1 if x == 'Y' else 0)
    df['OVERTIME'] = df['OVERTIME'].apply(lambda x: 1 if x == 'Yes' else 0)

    # One-hot encoding for categorical variables
    df = df.join(pd.get_dummies(df['BUSINESSTRAVEL'])).drop('BUSINESSTRAVEL', axis=1)
    df = df.join(pd.get_dummies(df['DEPARTMENT'], prefix='DEPARTMENT')).drop('DEPARTMENT', axis=1)
    df = df.join(pd.get_dummies(df['EDUCATIONFIELD'], prefix='EDUCATION')).drop('EDUCATIONFIELD', axis=1)
    df = df.join(pd.get_dummies(df['JOBROLE'], prefix='ROLE')).drop('JOBROLE', axis=1)
    df = df.join(pd.get_dummies(df['MARITALSTATUS'], prefix='STATUS')).drop('MARITALSTATUS', axis=1)

    # Apply map to each element
    df = df.map(lambda x: 1 if x is True else 0 if x is False else x)
    df = df.drop('EMPLOYEENUMBER', axis=1)
    df = df.drop(['EMPLOYEECOUNT', 'OVER18', 'STANDARDHOURS'], axis=1)


    # Model building and prediction
    X, y = df.drop('ATTRITION', axis=1), df['ATTRITION']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Training the model
    rf_model = RandomForestClassifier(n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Evaluating the model
    #score = rf_model.score(X_test, y_test)
    #stored_importances = dict(sorted(zip(rf_model.feature_names_in_, rf_model.feature_importances_), key=lambda x: x[1], reverse=True))

    # Preparing data that is to be predicted
    to_pred['GENDER'] = to_pred['GENDER'].apply(lambda x: 1 if x == 'Male' else 0)
    to_pred['OVER18'] = to_pred['OVER18'].apply(lambda x: 1 if x == 'Y' else 0)
    to_pred['OVERTIME'] = to_pred['OVERTIME'].apply(lambda x: 1 if x == 'Yes' else 0)

    # One-hot encoding for categorical variables
    to_pred = to_pred.join(pd.get_dummies(to_pred['BUSINESSTRAVEL'])).drop('BUSINESSTRAVEL', axis=1)
    to_pred = to_pred.join(pd.get_dummies(to_pred['DEPARTMENT'], prefix='DEPARTMENT')).drop('DEPARTMENT', axis=1)
    to_pred = to_pred.join(pd.get_dummies(to_pred['EDUCATIONFIELD'], prefix='EDUCATION')).drop('EDUCATIONFIELD', axis=1)
    to_pred = to_pred.join(pd.get_dummies(to_pred['JOBROLE'], prefix='ROLE')).drop('JOBROLE', axis=1)
    to_pred = to_pred.join(pd.get_dummies(to_pred['MARITALSTATUS'], prefix='STATUS')).drop('MARITALSTATUS', axis=1)

    # Apply map to each element
    to_pred = to_pred.map(lambda x: 1 if x is True else 0 if x is False else x)
    to_pred = to_pred.drop('EMPLOYEENUMBER', axis=1)
    to_pred = to_pred.drop(['EMPLOYEECOUNT', 'OVER18', 'STANDARDHOURS'], axis=1)

    # Predicting the attrition
    pred = rf_model.predict(to_pred)
    pred = pd.DataFrame(data = pred, columns = ['PREDICTED_ATTRITION'])
    pred = init_to_pred.join(pred)

    return pred
