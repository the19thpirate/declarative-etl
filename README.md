## Declarative ETL

- Are you a Data Scientist or a Data Analyst who is tired of writing code over and over again?
- You still wanna seem tech savvy but have an ease in your basic day to day analysis?
- Then this here is the tool for you


## How to use this tool
#### Currently only supports csv files.
1. Store your csv file in the data folder.
2. Create a virtual environment in your local machine and install the requirements (this tool is built on python3.8)
3. Head to the pipelines folder and create a .yaml file where you will spend most of your time.
4. You can specify the transformation steps in order of execution in this yaml file.
5. Please refer the existing default transformations made during development to get the format correct.
6. To run the pipeline simply run the following command: python reader.py {your_pipeline_name}.yaml
7. If you are running a data modelling pipeline then you will find the transformed dataset in the exports folder as a csv.
8. If you are running a regression model pipeline then you will find the model.pkl file in the model folder and its evaluation details inside the ./model/results folder.
