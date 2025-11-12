from abc import ABC, abstractmethod

class ModelPredictor(ABC):
    """ Tobi please change to whetever is suitable from the streamlit UI preprocessing,
        Additional_params is a dictionary which we can pass specific model params"""
    def __init__(self, name, time_window, granularity, additional_params=None):
        self.name = name
        self.time_window = time_window
        self.granularity = granularity
        self.additional_params = additional_params
        self._model = None #this will hold the model trained model object

    # pass the 3 datasets into this method
    def run(self, train_df, val_df, test_df):
        """All models follow this workflow of fit -> predict -> evaluate"""
        self._model = self.fit(train_df, val_df)
        predictions = self.predict(test_df)
        return self.evaluate(test_df, predictions)

    @abstractmethod
    def fit(self, train_df, val_df):
        pass
    
    @abstractmethod
    def predict(self, test_df):
        pass

    @abstractmethod
    def evaluate(self, test_df, predictions):
        pass

        
    


        
