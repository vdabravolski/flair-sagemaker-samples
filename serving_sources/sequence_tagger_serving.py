import json
import pickle
import logging

from flair.data import Sentence
from flair.models import SequenceTagger


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def model_fn(model_dir):
    """
    This method is called by Sagemaker when deploying hosting endpoint to deserialize model.
    model_dir is location where your trained model will be downloaded. in case of flair which automatically 
    downloads models from shared S3 bucket, we can ignore model_dir.
    
    Returns:
      - deserialized model object.
    """
    try:
        model = SequenceTagger.load('ner')
    
    except Exception as e:
        logger.error("Model deserialization failed.")
        logger.error(e)
        
    logger.debug("Model deserialization succesfully completed.")
    
    return model

def input_fn(request_body, request_content_type):
    """
    This method is called by Sagemaker for each inference request. It currently supports deserialized from "application/json" format only. 
    However, this can be easily extended. 
    
    Returns:
      - formatted data used for prediction.
    """
    
    if request_content_type.lower()=="application/json":
        inference_text = json.loads(request_body)
    else:
        raise ValueError(f"Format {request_content_type} is not supported. Please use \"application/json\" instead.")
    
    try:
        input_object = Sentence(inference_text)
        
    except Exception as e:
        logger.error("Converting inference text to FLAIR sentence failed.")
        logger.error(e)
    
    logger.debug("Input deserialization succesfully completed.")
    
    return input_object



def predict_fn(input_object, model):
    """
    Sagemaker calls predict_fn during inference. 
    
    Returns:
      - predictions in framework specific format.
    """
    
    try:
        prediction = model.predict(input_object)
        
    except Exception as e:
        logger.error("Prediction using FLAIR model failed.")
        logger.error(e)
    
    logger.debug(f"Model prediction: {str(prediction)}")
    
    return prediction

def output_fn(prediction, response_content_type):
    """
    Sagemaker calls output_fn to serialize predictions in desired format for end-user consumption.
    It currently supports pickling of FLAIR Sentence object.
    
    Returns:
      - serialized prediction object to send over wire.
    
    """
    
    if response_content_type.lower()=="pickle":
        
        try:
            pickled_output = pickle.dumps(prediction)
        except Exception as e:
            logger.error("Pickling of FLAIR sentence object failed.")
            logger.error(e)
            
    else:
        raise ValueError(f"Format {request_content_type} is not supported. Please use \"application/json\" instead.")
    
    logger.debug("Output serialization sucessfully completed. ")
    
    return pickled_output