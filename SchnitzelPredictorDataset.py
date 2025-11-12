import pandas as pd
import numpy as np

class SchnitzelPredictorDataset:
    def __init__(self, merged_dataset):
        self.merged_dataset = merged_dataset
        self.split = False

    def create_split_annotated_dataset(self, val_split_days, test_split_days):
        self.split_annotated_dataset = self._add_split_to_dataset(self.merged_dataset, val_split_days, test_split_days)
        self.split = True
    
    def get_split_annotated_dataset(self):
        if self.split == False:
            raise ValueError("Dataset not split yet.")
        
        return self.split_annotated_dataset
    
    def get_dataset(self):
        return self.merged_dataset

    def _add_split_to_dataset(self, merged_dataset, val_split_days, test_split_days):
        max_date = merged_dataset['DATE'].max()
        test_split_date = max_date - pd.Timedelta(days=test_split_days)
        val_split_date = test_split_date - pd.Timedelta(days=val_split_days)

        merged_dataset['SPLIT'] = np.where(
            merged_dataset['DATE'] >= test_split_date, 'TEST',
            np.where(
                merged_dataset['DATE'] >= val_split_date, 'VALIDATION', 'TRAIN'
            )
        )
        return merged_dataset

    def _get_grouped_dataset(self, grouping, dataset):
        if grouping not in ['ARTICLE', 'PRODUCT_GROUP', 'MAIN_GROUP']:
            raise ValueError("Invalid grouping.")
        
        return self._create_grouped_by_day(dataset, feature=grouping)
        

    def get_grouped_dataset(self, grouping):
        return self._get_grouped_dataset(grouping, dataset=self.merged_dataset)
        
        
    def get_grouped_dataset_by_split(self, grouping, split):
        if self.split == False:
            raise ValueError("Dataset not split yet.")
        
        dataset = self.get_split_annotated_dataset()
        dataset = dataset[dataset['SPLIT'] == split]

        return self._get_grouped_dataset(grouping, dataset=dataset)


    def _create_grouped_by_day(self, df_preprocessed, feature):
        df_grouped_by_day = df_preprocessed.groupby(['DATE', feature]).agg({
            'QUANTITY': 'sum',
        }).reset_index()

        return df_grouped_by_day
