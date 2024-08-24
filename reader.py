import yaml
import pandas as pd
from utils import pandas_transformations as pt
from utils import basic_ml as ml

with open("project.yaml") as instructions:
    try:
        instructionsList = yaml.safe_load(instructions).get("instructions")
    except Exception as e:
        print(e)

print(instructionsList)
processedData = pd.DataFrame()
for instruction in instructionsList:
    current_instruction = instruction.keys()
    if "data_loader" in current_instruction:
        processedData = pt.read_data(instruction)  
        print(processedData.columns)
    elif "sanitize_data" in current_instruction:
        processedData = pt.sanitize_headers(processedData) 
    elif "filter_data" in current_instruction:
        processedData = pt.filter_data(processedData, instruction)
    elif "group_data" in current_instruction:
        processedData = pt.group_data(processedData, instruction)
    elif "single_column_transformer" in current_instruction:
        processedData = pt.feature_engineer_single_col(processedData, instruction)
    elif "scale_data" in current_instruction:
        processedData = ml.scale_data(processedData, instruction)
    elif "split_data" in current_instruction:
        modelTrainDict = ml.build_train_test_split(processedData, instruction)
    elif "build_regression_model" in current_instruction:
        result = ml.build_regression(modelTrainDict, instruction)
    elif "data_store" in current_instruction:
        pt.save_data(processedData, instruction)
        print("Data Saved")

print(processedData.head(10))