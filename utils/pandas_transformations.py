import pandas as pd
import numpy as np
import math
import statistics

def read_data(instruction):
    location = instruction.get("data_loader")
    data = pd.read_csv(location)
    return data 

def group_data(data, instructions):
    group_data_dict = instructions.get("group_data")
    columnList = group_data_dict.get("columns")
    aggDict = group_data_dict.get("aggregations")
    groupedDf = data.groupby(columnList).agg(aggDict)
    groupedDf = groupedDf.reset_index().reset_index()
    return groupedDf

def sanitize_headers(data):
    columns = data.columns
    cleaned_columns = [text.lower().replace(" ", "_").strip() for text in columns]
    data.columns = cleaned_columns
    return data

def filter_data(data, instructions):
    column_name = instructions.get("column")
    column_type = data[column_name].dtype
    filter_value = instructions.get("filter_value") # Value to filter by
    operator = instructions.get("operator") # Operator
    if column_type == "object":
        data = data.query(f"{column_name} {operator} '{filter_value}'")
    else:
        data = data.query(f"{column_name} {operator} {filter_value}")
    return data

def feature_engineer_single_col(data, instructions):
    try:
        print(data.head())
        new_column_name = instructions.get("target_column_name")    
        column_name = instructions.get("column")
        formula_string = instructions.get("formula")
        formula_obj = solve_function(formula_string)
        new_column_data = formula_obj(data[column_name].to_numpy())
        if new_column_name == None:
            data[column_name] = new_column_data
        else:
            data[new_column_name] = new_column_data
        return data
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_lineno)

def solve_function(formula):
    def function(x):
        return eval(formula)
    return np.frompyfunc(function, 1, 1)

def save_data(data, instructions):
    return data.to_csv(instructions.get("data_store"))