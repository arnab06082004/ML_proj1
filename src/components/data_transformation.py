import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            num_cols = [ 'reading score', 'writing score']
            cat_cols = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']

            num_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('OneHot',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])

            logger.info(f'Numerical columns: {num_cols}')
            logger.info(f'Categorical columns: {cat_cols}')

            ct = ColumnTransformer([
                ('num_pipeline',num_pipeline,num_cols),
                ('cat_pipeline',cat_pipeline,cat_cols)
            ])

            return ct
            
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Read train and test data complete")

            preprocessing_obj = self.get_data_transformation_object()

            target_col = 'math score'
            num_cols = [ 'reading score', 'writing score']

            input_train_df = train_df.drop(columns=target_col, axis=1)
            target_train_df = train_df[target_col]

            input_test_df = test_df.drop(columns=target_col, axis=1)
            target_test_df = test_df[target_col]

            logger.info('applying preprocessing on train and test data')

            input_train_arr = preprocessing_obj.fit_transform(input_train_df)
            input_test_arr = preprocessing_obj.transform(input_test_df)

            train_arr = np.c_[input_train_arr,np.array(target_train_df)]
            test_arr = np.c_[input_test_arr,np.array(target_test_df)]

            logger.info('save preprocessing')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)