import pandas as pd

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
        return df_merged


    def create_grouped_by_day_and_article(self, df_preprocessed):
        # Convert DATE to datetime
        df_preprocessed['DATE'] = pd.to_datetime(df_preprocessed['DATE'], format='%d.%m.%y')

        # Group by DATE and aggregate
        df_grouped_by_day = df_preprocessed.groupby(['DATE', 'ARTICLE']).agg({
            'QUANTITY': 'sum',
        }).reset_index()

        return df_grouped_by_day
    
    def create_grouped_by_day_and_product_group(self, df_preprocessed):
        # Convert DATE to datetime
        df_preprocessed['DATE'] = pd.to_datetime(df_preprocessed['DATE'], format='%d.%m.%y')

        # Group by DATE and aggregate
        df_grouped_by_day = df_preprocessed.groupby(['DATE', 'PRODUCT_GROUP']).agg({
            'QUANTITY': 'sum',
        }).reset_index()

        return df_grouped_by_day
    
    def create_grouped_by_day_and_main_group(self, df_preprocessed):
        # Convert DATE to datetime
        df_preprocessed['DATE'] = pd.to_datetime(df_preprocessed['DATE'], format='%d.%m.%y')

        # Group by DATE and aggregate
        df_grouped_by_day = df_preprocessed.groupby(['DATE', 'MAIN_GROUP']).agg({
            'QUANTITY': 'sum',
        }).reset_index()

        return df_grouped_by_day