import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
import pyarrow.parquet as pq
from io import BytesIO
import boto3
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

print("***************************************************")

load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")




if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
    raise ValueError("AWS credentials or S3 bucket name not set")


s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

def read_and_merge_parquet_files(s3_bucket):
    try:
        
        objects = s3.list_objects(Bucket=s3_bucket)

       
        dfs = []
        for obj in objects.get('Contents', []):
            
            response = s3.get_object(Bucket=s3_bucket, Key=obj['Key'])
            parquet_file = BytesIO(response['Body'].read())
            df = pq.read_table(parquet_file).to_pandas()
            dfs.append(df)

       
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            return merged_df
        else:
            print("No Parquet files found in the S3 bucket.")
            return None
    
    except Exception as e:
        print(f"Error: {e}")
        return None


merged_data = read_and_merge_parquet_files(S3_BUCKET_NAME)


if merged_data is not None:
    print(merged_data)
else:
    print("Error occurred during processing.")

        




print("***************************************************")



merged_data.replace('na', np.nan, inplace=True)



merged_data['class'] = merged_data['class'].replace('pos', 1).replace(['neg'], 0)



merged_data.drop('partition_0',axis=1,inplace=True)


x = [i for i in merged_data.columns]
filtered_list = [item for item in x if item not in ("class")]




for column in filtered_list:
    merged_data[column] = pd.to_numeric(merged_data[column], errors='coerce')


correlation_matrix = merged_data.corr()
print("***************************************************")

highly_correlated_pairs = []
considered_columns = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        colname1 = correlation_matrix.columns[i]
        colname2 = correlation_matrix.columns[j]
        
      
        if colname1 != colname2:
   
            if colname1 not in considered_columns and colname2 not in considered_columns:
                if abs(correlation_matrix.iloc[i, j]) > 0.85:
                    pair = (colname1, colname2)
                    highly_correlated_pairs.append(pair)
                    considered_columns.add(colname1)
                    considered_columns.add(colname2)

unique_columns = list(set(column for pair in highly_correlated_pairs for column in pair))

merged_data.drop(unique_columns,axis=1,inplace=True)
print("***************************************************")

X_train, X_test, y_train, y_test = train_test_split(merged_data.drop('class', axis=1) , merged_data['class'], test_size=0.2, random_state=42)

print("***************************************************")

numeric_features=X_train.columns

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=6)) 
        ]), numeric_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

X_train = pipeline.fit_transform(X_train)
X_test= pipeline.transform(X_test)

print("***************************************************")

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

params = {
    'booster': ['dart'],
    'colsample_bytree': [0.8520316243427525],
    'eval_metric': ['logloss'],
    'gamma': [0.2671804347792823],
    'learning_rate': [0.04548227865598075],
    'max_depth': [10],
    'objective': ['binary:logistic'],
    'subsample': [0.6510467984216483]
}

xgb_model = XGBClassifier()

grid_search = GridSearchCV(xgb_model, param_grid=params, cv=3, scoring='recall')

grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)
print("***************************************************")

y_pred = grid_search.predict(X_test)

y_pred_binary = (y_pred > 0.5).astype(int)  
accuracy = accuracy_score(y_test, y_pred_binary)

print("Accuracy on the test set:", accuracy)

output_directory = 'Aux_data'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


conf_matrix = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.savefig('Aux_data/confusion_matrix.png')
#plt.show()


base_directory = 'Model'
if not os.path.exists(base_directory):
    os.makedirs(base_directory)


local_path_p1 = os.path.join(base_directory, 'pipeline.joblib')
joblib.dump(pipeline, local_path_p1)


local_path_p2 = os.path.join(base_directory, 'xgboost.joblib')
joblib.dump(grid_search, local_path_p2)

print("Pipelines saved successfully locally.")
