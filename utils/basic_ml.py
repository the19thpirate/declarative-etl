from sklearn import metrics
import pandas as pd
import numpy as np
import warnings as wr
import pickle
wr.filterwarnings("ignore")

# Scaling Module
def scale_data(data, instructions):
    ## Current available options: StandardScaler, MinMaxScaler
    scale_type = instructions.get("scaler_type")
    columns = instructions.get("columns")
    merge_dataset = instructions.get("merge_dataset")
    if columns == None:
        # We will only choose numerical columns if no columns are provided
        columns = [column for column in data.columns if data[column].dtype in (int, float)]

    subset = data[columns]
    if scale_type == "min_max":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    
    scaled_data = scaler.fit_transform(subset) 
    scaled_data = pd.DataFrame(scaled_data, columns = columns)

    # Append scaled numerical data with original dataset
    if merge_dataset == True:
        data.drop(columns, axis = 1, inplace = True)
        data['id'] = np.arange(data.shape[0])
        scaled_data['id'] = np.arange(data.shape[0])
        data = data.merge(scaled_data, on = "id")
        data.drop("id", axis = 1, inplace = True)
    else:
        data = scaled_data
    return data


# Train Test Split Module
def build_train_test_split(data, instructions):
    test_size = instructions.get("test_size")
    random_state = instructions.get("random_state", 1) # If not provided we set it to 1
    select_columns = instructions.get("not_targets")
    target_column = instructions.get("target")
    X = data[select_columns]
    y = data[target_column]

    # encoding the categorical columns 
    for column in X.columns:
        if X[column].dtype == "object":
            X[column] = pd.Categorical(X[column]).codes
        else:
            continue

    from sklearn.model_selection import train_test_split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=test_size, random_state=random_state)

    model_train_dict = {
        "training_data" : [X_train, y_train],
        "testing_data" : [X_test, y_test],
        "validation_data" : [X_val, y_val]
    }
    # We return a dictionary containing the required data.
    return model_train_dict


# Regression Module
# Should allow the user to save the model as a pickle file
def build_regression(data_dict, instructions):
    try:
        hyper_parameters = instructions.get("hyper_parameters")
        training_data = data_dict.get("training_data")
        testing_data = data_dict.get("testing_data")
        model_name = data_dict.get("model_type", "model")
        validation_data = data_dict.get("validation_data")
        X_train, y_train = training_data[0], training_data[1]
        X_test, y_test = testing_data[0], testing_data[1]
        X_val, y_val = validation_data[0], validation_data[1]

        if model_name == "RandomForestRegressor":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor()
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor()

        print(X_train.shape, y_train.shape)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        rmse = np.sqrt(metrics.mean_squared_error(y_val, val_pred))
        r2_score = metrics.r2_score(y_val, val_pred)
        mae = metrics.mean_absolute_error(y_val, val_pred)
        validation_result = pd.DataFrame(
            [{
                'RMSE' : rmse, 'R2' : r2_score, "MAE" : mae
            }]
        )
        
        model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        y_pred = model.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2_score = metrics.r2_score(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)

        testing_result = pd.DataFrame(
            [{
                'RMSE' : rmse, 'R2' : r2_score, "MAE" : mae
            }]
        )

        ## Saving the assets
        validation_result.to_csv(f"./model/results/validation_result.csv")
        testing_result.to_csv(f"./model/results/testing_result.csv")

        with open(f"./model/{model_name}.pkl", "wb") as f:
            pickle.dump(model, f)

        return "Success"
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_lineno)

# Simple Logistic Regression Module 
    