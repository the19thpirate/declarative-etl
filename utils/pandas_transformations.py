import pandas as pd
import numpy as np
import re
import math
import statistics

def read_data(instruction):
    instruction_dict = instruction.get("data_loader")
    location = instruction_dict.get("file_path")
    sanitize_data = instruction_dict.get("sanitize_data")
    data = pd.read_csv(location)
    if sanitize_data == True:
        data = sanitize_headers(data)
    return data 

def data_information(data):
    return data.info()

def data_description(data, instruction):
    percentile_list = instruction.get("percentile_list")
    describe_data = data.describe(
        percentiles = percentile_list
    )
    return describe_data

def fix_datatypes(data, instruction):
    columns_dict = instruction.get("column_mapping")
    
    for column, datatype in columns_dict.items():
        data[column] = data[column].astype(datatype)
    
    return data

def sanitize_headers(data):
    columns = data.columns
    cleaned_columns = [text.lower().replace(" ", "_").strip() for text in columns]
    data.columns = cleaned_columns
    return data

def group_data(data, instructions):
    group_data_dict = instructions.get("group_data")
    columnList = group_data_dict.get("columns")
    aggDict = group_data_dict.get("aggregations")
    groupedDf = data.groupby(columnList).agg(aggDict)
    groupedDf = groupedDf.reset_index().reset_index()
    return groupedDf

def filter_data(data, instructions):
    instruction_dict = instructions.get("filter_data")
    column_name = instruction_dict.get("column")
    column_type = data[column_name].dtype
    filter_value = instruction_dict.get("filter_value") # Value to filter by
    operator = instruction_dict.get("operator") # Operator
    if column_type == "object":
        data = data.query(f"{column_name} {operator} '{filter_value}'")
    else:
        data = data.query(f"{column_name} {operator} {filter_value}")
    return data

def feature_engineer_single_col(data, instructions):
    try:
        print(data.head())
        instructions_dict = instructions.get("single_column_transformer")
        new_column_name = instructions_dict.get("target_column_name")    
        column_name = instructions_dict.get("column")
        formula_string = instructions_dict.get("formula")
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

def find_regex_string(data, instructions):
    instruction_dict = instructions.get("regex_search")
    regex_pattern = instruction_dict.get("pattern")
    regex_column = instruction_dict.get("column")
    target_column_name = instruction_dict.get("target_column_name")
    if target_column_name == None:
        target_column_name = regex_column + '_exists_flag'
    
    data[target_column_name] = data[regex_column].apply(
        lambda x: True if re.search(regex_pattern, x.lower().strip()) else False
    )   
    return data

def save_data(data, instructions):
    instructions_dict = instructions.get("data_store")
    store_path = instructions_dict.get("path")
    return data.to_csv(store_path)