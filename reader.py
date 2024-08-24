import yaml
import pandas as pd
from utils import pandas_transformations as pt
from utils import basic_ml as ml
from utils import visualize as viz
import sys
pd.set_option('display.max_columns', None)


total_args = len(sys.argv)
if total_args > 2:
    print("Too many arguments have been passed. Only specify one pipeline at a time")
    sys.exit()
else:
    file_name = sys.argv[0]
    pipe_name = sys.argv[1]
    print(f"Argument count okay!\nRunning Pipe: {pipe_name}")


with open(f"./pipelines/{pipe_name}") as instructions:
    try:
        instructionsList = yaml.safe_load(instructions).get("instructions")
    except Exception as e:
        print(e)

print(instructionsList)
for instruction in instructionsList:
    current_instruction = instruction.keys()
    if "data_loader" in current_instruction:
        processedData = pt.read_data(instruction)  
    elif "view_data" in current_instruction:
        print(processedData.head())
    elif "data_description" in current_instruction:
        print(pt.data_description(processedData, instruction))
    elif "data_information" in current_instruction:
        print(processedData.info())
    elif "fix_datatype" in current_instruction:
        processedData = pt.fix_datatypes(processedData, instruction)
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
        processedData = ml.build_regression(modelTrainDict, instruction)
    elif "univariate_analysis" in current_instruction:
        result = viz.univariate_analysis(processedData, instruction)
    elif "bivariate_analysis" in current_instruction:
        result = viz.bi_variate_analysis(processedData, instruction)
    elif "data_store" in current_instruction:
        pt.save_data(processedData, instruction)
        print("Data Saved")