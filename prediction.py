import os
import joblib
import pandas as pd
import sys
import joblib
import numpy as np

def main():
   
    df = pd.read_csv('Temp_data/your_data.csv')  
   
    df.replace('na', np.nan, inplace=True)
    


    with open('Aux_data/column_required.txt', 'r') as file:
        columns_list = [line.strip() for line in file.readlines()]
    

    try:
       
        all_columns_present = all(column in df.columns for column in columns_list)

 
        num_columns = len(df.columns)

       
        if all_columns_present and num_columns == len(columns_list):
            print("All columns in l1 are present in the DataFrame df.")
            print("Number of columns in df:", num_columns)
        else:
            raise ValueError("Not all columns in l1 are present in the DataFrame df, or the number of columns is different.")

    except ValueError as e:
        print(e)
        sys.exit("Error: Program stopped due to unmet conditions.")


    model_dir = 'model'

  
    p1 = joblib.load(os.path.join(model_dir, 'pipeline.joblib'))
    xgb = joblib.load(os.path.join(model_dir, 'xgboost.joblib'))




    transformed_data = p1.transform(df)


    predictions = xgb.predict(transformed_data)
    df['prediction'] = predictions

    print(df)


 
if __name__ == "__main__":
    main()
