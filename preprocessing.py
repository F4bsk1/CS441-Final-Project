import pandas as pd
import numpy as np
from SchnitzelPredictorDataset import SchnitzelPredictorDataset

class Preprocessor:
    def __init__(self, df_transaction_details, df_transaction_header, df_product_groups, df_articles):
        self.df_transaction_details = df_transaction_details
        self.df_transaction_header = df_transaction_header
        self.df_product_groups = df_product_groups
        self.df_articles = df_articles

    def preprocess(self):
        
        #Clean product groups and articles
        self.df_articles.columns = self.df_articles.columns.str.replace('"', '')
        self.df_product_groups.columns = self.df_product_groups.columns.str.replace('"', '')
        #Clean transaction details
        #tbd



        #Remove irrelevant columns
        self.df_transaction_header = self.df_transaction_header.drop(['ECR_NO', 'DOC_NO', 'DOC_TYPE', "DOC_SOURCE", "NOT_PRINTED", "CLERK_NO", "CLERK_NAME", "TRAIN_MODE", "CURR_ISO_NO", "SALES_EXCL_VAT", "BOOK_MEM_TYPE", "BOOK_MEM_NO"], axis=1)
        
        # Rename columns
        self.df_transaction_details.rename(columns={"DOC_KEY_LINK": "DOC_KEY", "BASE_NO": "SKU", "BASE_NAME": "ARTICLE_IN_TRANSACTION", "DEPT_NO":"PRODUCT_GROUP_NO", "DEPT_NAME": "PRODUCT_GROUP_IN_TRANSACTION"}, inplace=True)
        self.df_transaction_header.rename(columns={"DOC_DATE": "DATE", "DOC_TIME": "TIME"}, inplace=True)
        self.df_articles.rename(columns={"Nr": "SKU", "Name": "ARTICLE", "Warengruppe": "PRODUCT_GROUP_NO"}, inplace=True)
        self.df_product_groups.rename(columns={"Nr": "PRODUCT_GROUP_NO", "Name": "PRODUCT_GROUP", "Hauptgruppe": "MAIN_GROUP_NO", "Hauptgruppe Name": "MAIN_GROUP"}, inplace=True)
        #return self.df_articles

        #Merge ds
        print(self.df_transaction_details.columns)
        print(self.df_transaction_details.head())
        df_merged = pd.merge(self.df_transaction_details, self.df_transaction_header, on='DOC_KEY', how='left')
        df_merged['HOUR'] = pd.to_datetime(df_merged['TIME'], format='%H:%M:%S').dt.hour
        print(df_merged.columns)
        print(df_merged.head())
        df_merged = df_merged.groupby(['DATE', 'HOUR', 'SKU'], as_index=False)['QUANTITY'].sum()
        df_merged['DATE'] = pd.to_datetime(df_merged['DATE'], format='%d.%m.%y')

        print(df_merged.head())
        #fill skus for each day
        all_dates = pd.date_range(df_merged['DATE'].min(), df_merged['DATE'].max())
        all_skus = df_merged['SKU'].unique()
        all_hours = range(0,24)
        all_combinations = pd.MultiIndex.from_product([all_dates, all_hours, all_skus], names=['DATE', 'HOUR', 'SKU']).to_frame(index=False)
        df_ext = pd.merge(all_combinations, df_merged, on=['DATE', 'HOUR', 'SKU'], how='left')
        df_ext['QUANTITY'] = df_ext['QUANTITY'].fillna(0)
        df_ext = df_ext.sort_values(['DATE', 'HOUR', 'SKU']).reset_index(drop=True)
        print(df_ext.head())

        #Merge products groups and articles in
        df_merged = pd.merge(df_ext, self.df_articles, on='SKU', how='left')
        print(df_merged.columns)
        df_merged = pd.merge(df_merged, self.df_product_groups, on='PRODUCT_GROUP_NO', how='left')

        df_merged['YEAR'] = df_merged['DATE'].dt.year
        df_merged['MONTH'] = df_merged['DATE'].dt.month
        df_merged['DAY'] = df_merged['DATE'].dt.day
        df_merged['DAYOFWEEK'] = df_merged['DATE'].dt.dayofweek
        df_merged['WEEKOFYEAR'] = df_merged['DATE'].dt.isocalendar().week.astype(int)
        df_merged['IS_WEEKEND'] = df_merged['DAYOFWEEK'].isin([5,6]).astype(int)
        #df_merged = df_merged.drop(columns=['DATE'])

        schnitzelPredictorDataset = SchnitzelPredictorDataset(df_merged)

        return schnitzelPredictorDataset


