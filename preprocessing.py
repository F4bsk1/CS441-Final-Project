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
        self.df_articles.rename(columns={"Nr": "SKU", "Name": "ARTICLE", "Warengruppe": "PRODUCT_GROUP_NO_ARTICLES"}, inplace=True)
        self.df_product_groups.rename(columns={"Nr": "PRODUCT_GROUP_NO", "Name": "PRODUCT_GROUP", "Hauptgruppe": "MAIN_GROUP_NO", "Hauptgruppe Name": "MAIN_GROUP"}, inplace=True)
        #return self.df_articles

        #Merge ds
        df_merged = pd.merge(self.df_transaction_details, self.df_transaction_header, on='DOC_KEY', how='outer')
        df_merged = pd.merge(df_merged, self.df_articles, on='SKU', how='left')
        df_merged = pd.merge(df_merged, self.df_product_groups, on='PRODUCT_GROUP_NO', how='left')

        df_merged['DATE'] = pd.to_datetime(df_merged['DATE'], format='%d.%m.%y')

        schnitzelPredictorDataset = SchnitzelPredictorDataset(df_merged)

        return schnitzelPredictorDataset


