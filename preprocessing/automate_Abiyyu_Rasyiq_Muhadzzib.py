import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, RobustScaler
from imblearn.over_sampling import SMOTE


def preprocess_data(df, target_column='loan_status', test_size=0.2, random_state=42):    
    # 1. Handling Missing & Duplicates
    df_clean = df.dropna().drop_duplicates()
    
    # 2. Split Data
    X = df_clean.drop(target_column, axis=1)
    y = df_clean[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )
    
    # 3. Outlier Treatment
    X_train['person_age'] = X_train['person_age'].clip(upper=80)
    X_train['person_emp_exp'] = X_train['person_emp_exp'].clip(upper=50)
    upper_limit_income = X_train['person_income'].quantile(0.99)
    X_train['person_income'] = X_train['person_income'].clip(upper=upper_limit_income)
    
    # 4. Encoding
    # One-Hot Encoding
    cols_onehot = ['person_home_ownership', 'loan_intent']
    onehot_encoder = OneHotEncoder(sparse_output=False, drop=None)
    X_train_encoded = onehot_encoder.fit_transform(X_train[cols_onehot])
    X_test_encoded = onehot_encoder.transform(X_test[cols_onehot])
    
    encoded_cols = onehot_encoder.get_feature_names_out(cols_onehot)
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)
    
    X_train_final = pd.concat([X_train.drop(columns=cols_onehot), X_train_encoded_df], axis=1)
    X_test_final = pd.concat([X_test.drop(columns=cols_onehot), X_test_encoded_df], axis=1)
    
    # Label Encoding
    cols_binary = ['person_gender', 'previous_loan_defaults_on_file']
    label_encoders = {}
    for col in cols_binary:
        le = LabelEncoder()
        X_train_final[col] = le.fit_transform(X_train_final[col])
        X_test_final[col] = le.transform(X_test_final[col])
        label_encoders[col] = le
    
    # Ordinal Encoding
    education_order = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
    ordinal_encoder = OrdinalEncoder(categories=[education_order])
    X_train_final['person_education'] = ordinal_encoder.fit_transform(X_train_final[['person_education']])
    X_test_final['person_education'] = ordinal_encoder.transform(X_test_final[['person_education']])
    
    # 5. SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_final, y_train)
    
    # 6. Scaling
    numeric_cols = ['person_age', 'person_income', 'person_emp_exp', 
                    'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                    'cb_person_cred_hist_length']
    
    scaler = RobustScaler()
    X_train_resampled[numeric_cols] = scaler.fit_transform(X_train_resampled[numeric_cols])
    X_test_final[numeric_cols] = scaler.transform(X_test_final[numeric_cols])
    
    # Fitted objects untuk inference
    fitted_objects = {
        'onehot_encoder': onehot_encoder,
        'label_encoders': label_encoders,
        'ordinal_encoder': ordinal_encoder,
        'scaler': scaler,
        'upper_limit_income': upper_limit_income,
        'numeric_cols': numeric_cols,
        'encoded_cols': list(encoded_cols)
    }
    
    return X_train_resampled, X_test_final, y_train_resampled, y_test, fitted_objects