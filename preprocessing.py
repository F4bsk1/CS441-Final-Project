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
        #self.df_transaction_header = self.df_transaction_header.drop(['TTL_VAT_INCL', 'TTL_VAT_EXCL', 'TTL_EXPENSES', 'TTL_TIP', 'TTL_DECL', 'TTL_DIFF', 'TTL_DRAWER', 'TTL_PETTYCASH', 'TAX_1_RATE	TAX_1_TTL', 'TAX_1_TTL_NEG', 'TAX_1_VAL', 'TAX_1_MAPPING', 'TAX_2_RATE', 'TAX_2_TTL', 'TAX_2_TTL_NEG', 'TAX_2_VAL', 'TAX_2_MAPPING', 'TAX_3_RATE', 'TAX_3_TTL', 'TAX_3_TTL_NEG', 'TAX_3_VAL', 'TAX_3_MAPPING', 'TAX_4_RATE', 'TAX_4_TTL', 'TAX_4_TTL_NEG', 'TAX_4_VAL', 'TAX_4_MAPPING', 'TAX_5_RATE', 'TAX_5_TTL', 'TAX_5_TTL_NEG', 'TAX_5_VAL', 'TAX_5_MAPPING', 'TAX_6_RATE', 'TAX_6_TTL	TAX_6_TTL_NEG', 'TAX_6_VAL', 'TAX_6_MAPPING', 'TAX_7_RATE', 'TAX_7_TTL', 'TAX_7_TTL_NEG', 'TAX_7_VAL', 'TAX_7_MAPPING', 'TAX_8_RATE	TAX_8_TTL', 'TAX_8_TTL_NEG', 'TAX_8_VAL', 'TAX_8_MAPPING', 'TAX_0_RATE', 'TAX_0_TTL', 'TAX_0_TTL_NEG', 'TAX_0_VAL', 'TAX_0_MAPPING', 'BLK_KEY_LINK', 'REG_KEY_LINK', 'REG_START_DATE', 'ECR_NO_SUPPL', 'POS_ID_NO', 'CERT_SERIAL_NO	AUTH_CLERK_NO', 'AUTH_CLERK_NAME', 'CONTACT_KEY_LINK', 'SIGNATURE', 'AUTO_VOID'])
        #self.df_transaction_details = self.df_transaction_details.drop(['AMOUNT	AMOUNT_LINK	TAX_1_TTL', 'TAX_2_TTL', 'TAX_3_TTL', 'TAX_4_TTL', 'TAX_5_TTL', 'TAX_6_TTL', 'TAX_7_TTL', 'TAX_8_TTL', 'TAX_0_TTL', 'OVERWRITE', 'TTL_MOD', 'VOID', 'ECR_NO', 'ECR_NO_SUPPL', 'BLK_KEY_LINK', 'TTL_PRICE_SHIFT', 'ORIG_PRICE_PRICESHIFT', 'TTL_OVERWRITE', 'ORIG_PRICE_OVERWRITE', 'TTL_DISCOUNT', 'NON_ADD_NUMBER'])
        print(f"trans_header: {self.df_transaction_header.columns}")
        print(f"trans_details: {self.df_transaction_details.columns}")
        # Rename columns
        self.df_transaction_details.rename(columns={"DOC_KEY_LINK": "DOC_KEY", "BASE_NO": "SKU", "BASE_NAME": "ARTICLE_IN_TRANSACTION", "DEPT_NO":"PRODUCT_GROUP_NO", "DEPT_NAME": "PRODUCT_GROUP_IN_TRANSACTION"}, inplace=True)
        self.df_transaction_header.rename(columns={"DOC_DATE": "DATE", "DOC_TIME": "TIME"}, inplace=True)
        self.df_articles.rename(columns={"Nr": "SKU", "Name": "ARTICLE", "Warengruppe": "PRODUCT_GROUP_NO"}, inplace=True)
        self.df_product_groups.rename(columns={"Nr": "PRODUCT_GROUP_NO", "Name": "PRODUCT_GROUP", "Hauptgruppe": "MAIN_GROUP_NO", "Hauptgruppe Name": "MAIN_GROUP"}, inplace=True)
        #return self.df_articles
    
        #Merge ds
        print(self.df_transaction_details.columns)
        print(self.df_transaction_details.head())
        print(self.df_transaction_details['QUANTITY'].unique())
        df_merged = pd.merge(self.df_transaction_details, self.df_transaction_header, on='DOC_KEY', how='left')
        df_merged['QUANTITY'] = pd.to_numeric(df_merged['QUANTITY'], errors='coerce')
        df_merged = df_merged[df_merged['QUANTITY'].notna()]  # remove NaN
        df_merged = df_merged[df_merged['QUANTITY'] >= 0]     # remove negatives
        df_merged = df_merged[df_merged['QUANTITY'] <= 1000]


        df_merged['HOUR'] = pd.to_datetime(df_merged['TIME'], format='%H:%M:%S').dt.hour
        print("before group by in preprocess")
        print(df_merged.columns)
        print(df_merged.head())
        print(df_merged['QUANTITY'].unique())
        df_merged = df_merged.groupby(['DATE', 'HOUR', 'SKU'], as_index=False)['QUANTITY'].sum()
        print(df_merged.head())
        if len(df_merged['DATE'][0]) == 10:
            df_merged['DATE'] = pd.to_datetime(df_merged['DATE'], format='%d.%m.%Y')
        else:
            df_merged['DATE'] = pd.to_datetime(df_merged['DATE'], format='%d.%m.%y')

        df_merged = df_merged[df_merged['DATE'] <= pd.to_datetime('01.12.2024', format='%d.%m.%Y')]

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
        #df_merged['DAYOFWEEK'] = df_merged['DATE'].dt.dayofweek
        #df_merged['WEEKOFYEAR'] = df_merged['DATE'].dt.isocalendar().week.astype(int)
       # df_merged['IS_WEEKEND'] = df_merged['DAYOFWEEK'].isin([5,6]).astype(int)
        #df_merged = df_merged.drop(columns=['DATE'])

        #create timelags
        #df_merged = df_merged[df_merged['MAIN_GROUP'] == "Main Dishes"] 
        
        schnitzelPredictorDataset = SchnitzelPredictorDataset(df_merged)

        return schnitzelPredictorDataset


