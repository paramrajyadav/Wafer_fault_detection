# Wafer Fault Detection Project



## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Step 1: Clone the Repository](#step-1-clone-the-repository)
  - [Step 2: Install Dependencies](#step-2-install-dependencies)
  - [Step 3: Run ETL Process](#step-3-run-etl-process)
  - [Step 4: Train the Model](#step-4-train-the-model)
  - [Step 5: Start the Streamlit App](#step-5-start-the-streamlit-app)
- [Libraries Used](#libraries-used)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The Wafer Fault Detection project aims to develop a robust machine learning pipeline for classifying wafer faults, with a specific focus on issues related to the pneumatic braking system. 
The primary objective is to enhance the efficiency and reliability of fault detection in wafer manufacturing processes.

## Getting Started

### Prerequisites

List any prerequisites that users need to have installed before they can use your project.

- [Python](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)

### Installation

Below are the steps required for installation

Step 1: Clone the Repository

```bash
git clone https://github.com/paramrajyadav/wafer_fault_detection.git
cd yourproject
```

Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Step 3: Train the Model
```bash
python train_model.py
```
Step 4: Start the Streamlit App
```bash
streamlit run app.py
```


# Libraries Used  

Here are the key libraries and dependencies used in this project:

- [NumPy](https://numpy.org/): Used for numerical operations and array manipulation in Python.

- [Pandas](https://pandas.pydata.org/): Utilized for data manipulation and analysis.

- [Scikit-learn](https://scikit-learn.org/): Employed for machine learning algorithms and model evaluation.

- [XGBoost](https://xgboost.readthedocs.io/): Used for implementing the XGBoost machine learning algorithm.

- [MLflow](https://www.mlflow.org/): Integrated for managing the end-to-end machine learning lifecycle, including experimentation, reproducibility, and deployment.

- [Optuna](https://optuna.readthedocs.io/): Utilized for hyperparameter optimization, enhancing model performance.

- [Streamlit](https://streamlit.io/): Employed for building interactive web applications for data science and machine learning.

- [AWS SDK for Python (Boto3)](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html): Used for interacting with Amazon Web Services.





## Project Responsibilities  

As a contributor to this project, the following responsibilities were undertaken:

Conducted Exploratory Data Analysis (EDA)
Fine-tuned hyperparameters for machine learning models
Selected appropriate classification models
Trained models for fault detection
Utilized GitAction, Docker, AWS EC2, AWS ECR and Streamlit Cloud for deployment
Machine Learning Methods
Various machine learning techniques were employed, with a focus on classification models suitable for fault detection. 
Key algorithms included XGBoost, Random Forest, and Decision Tree. Hyperparameter fine-tuning was performed using and Optuna to optimize model performance.

## Continuous Integration and Continuous Delivery (CI/CD)  
To ensure an efficient development and deployment process, GitAction was utilized for automating the CI/CD pipeline. This automation guarantees that the model is consistently updated and can be deployed swiftly to production environments.

## Data Sources and ETL Workflow  
Data retrieval was accomplished from an S3 bucket, and AWS Lambda along with AWS Glue was employed for Extract, Transform, Load (ETL) processes. This ETL workflow facilitated the conversion of raw data into a format suitable for training and evaluating machine learning models, specifically tailored to detect faults related to the pneumatic braking system in wafer manufacturing.

## Deployment  
The model has been deployed to a Streamlit app, allowing users to interact with the model and observe its predictions. 
The deployed app is accessible at https://waferfaultdetection.streamlit.app/.

## Project Links  
Streamlit App: https://waferfaultdetection.streamlit.app/


For questions, suggestions, or issues related to the project, you can contact the author:

**Author:** Raj Kumar Yadav  
**Email:** paramrajyadav@gmail.com  
**GitHub:** [paramrajyadav](https://github.com/paramrajyadav)

Feel free to reach out for any inquiries or collaboration opportunities!






