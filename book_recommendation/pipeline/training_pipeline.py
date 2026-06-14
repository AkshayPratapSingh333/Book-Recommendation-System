from book_recommendation.components.step_01_data_ingestion import DataIngestion
from book_recommendation.components.step_02_data_validation import DataValidation



class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()


    def start_training_pipeline(self):
        """
        Starts the training pipeline
        :return: none
        """
        self.data_ingestion.initiate_data_ingestion()
        self.data_validation.initiate_data_validation()