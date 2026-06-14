from book_recommendation.components.step_01_data_ingestion import DataIngestion
from book_recommendation.components.step_02_data_validation import DataValidation
from book_recommendation.components.step_03_data_transformation import DataTransformation



class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()
        self.data_transformation = DataTransformation()


    def start_training_pipeline(self):
        """
        Starts the training pipeline
        :return: none
        """
        self.data_ingestion.initiate_data_ingestion()
        self.data_validation.initiate_data_validation()
        self.data_transformation.initiate_data_transformation()