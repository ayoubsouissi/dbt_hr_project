import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def model(dbt, session):
    dbt.config(
        packages=["pandas", "scikit-learn"]
    )

    # Load the data
    df = dbt.ref("stg_hr_actuals").to_pandas()

    return df