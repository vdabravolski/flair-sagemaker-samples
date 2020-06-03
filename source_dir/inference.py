__authors__ = ['Ibrahim Gabr', 'Vadim Dabravolski']

import json
import pickle
import logging
import sys
import os
import torch

from flair.data import Sentence

JSON_CONTENT_TYPE = 'application/json'
CSV_CONTENT_TYPE = 'text/csv'
PICKLE_CONTENT_TYPE = 'pickle'


# Ensure logging to /logs/mms_logs.log in the container.
# Logging everything from INFO level and above
# Stream output to stdout -> will be visible in docker container stdout in local mode and in cloud watch when using dedicated endpoints.
logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)

## NOTE:
"""
The methods input_fn and output_fn are OPTIONAL.
If obmitted SageMaker will assume:

the input and output objects are of type NPY format with Content-Type application/x-npy.
"""

def model_fn(model_dir):
    """
    A function to load up your model into memory.

    The return of this function is passed to predict_fn at the time of inference.

    This function is only run once at the start if the container/endpoint.

    Returns:
      - model object. Return value of this function is passed to predict_fn
    """

    # Boilerplate code - you dont need to worry about model_dir
    # Ensure that your model filename matches what you placed in the tar.gz
    try:
        with open(os.path.join(model_dir, 'flair_model.pth'), 'rb') as f:
            model = torch.load(f)
        
        logging.info("Model loaded into memory")
        
        return model
    except Exception as e:
        logging.exception("Model could not be loaded")


# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=JSON_CONTENT_TYPE):
    """
    This method is called by SageMaker for every inference request. It handles the INPUT DATA sent to the model/endpoint.

    This method needs to deserialze the invoke request body into an object we can perform prediction on.
    
    It currently natively supports serialized form "application/json" format only. 
   
    However, this can be easily extended as show below (i.e. text/csv)
    
    Returns:
      - formatted data used for prediction. The return value of this function is passed to predict_fn
    """
    
    # If the request is submitted/serialized as application/json
    if content_type.lower() == JSON_CONTENT_TYPE:
        inference_text = json.loads(request_body)
    
    # If the request is submitted/serialized as text/csv
    elif content_type.lower() == CSV_CONTENT_TYPE:
        inference_text=request_body
    
    else:
        raise ValueError(f"Format {content_type} is not supported. Please use one of {[JSON_CONTENT_TYPE, CSV_CONTENT_TYPE]}")
    
    # Turning the request_body into a FLAIR sentence
    try:
        input_object = Sentence(inference_text)
        
    except Exception as e:
        logging.exception("Converting inference text to FLAIR sentence failed.")
    
    logging.info("Input serialization succesfully completed.")
    
    return input_object


# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    """
    This function is called by SageMaker for every inference request.
    
    This method takes the deserialized request object and performs inference against the loaded model.
    
    Returns:
      - predictions in framework specific format.
    """
    # defaults to cuda:0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model.to(device) # ensuring your model is loaded on the approriate hardware

        model.eval() # model.eval() will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.

        with torch.no_grad(): # torch.no_grad() impacts the autograd engine and deactivates it. It will reduce memory usage and speed up computations

            prediction = model.predict(input_object)

        logging.info(f"Model prediction: {str(prediction)}")
        
    except Exception as e:
        logging.exception("Inference failed.")
    
    return prediction

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=PICKLE_CONTENT_TYPE):
    """

    This function is called by SageMaker for every inference request to serialize predictions in desired format for end-user consumption.

    This method takes the result of prediction and serializes this according to the response content type (accept).

    For our purposes, we will pickle the response of type flair.data.Sentence
    
    Returns:
      - serialized prediction object to send back to client.
    """
    
    if accept.lower() == PICKLE_CONTENT_TYPE:
        try:
            output = pickle.dumps(prediction)
        except Exception as e:
            logging.exception("Pickling of FLAIR sentence object failed.")
    
    elif accept.lower() == CSV_CONTENT_TYPE:
        output = str(prediction)

    # Note how you can add additional logic here - i.e. json.dumps({result: str(prediction)})
    elif accept.lower() == JSON_CONTENT_TYPE:
        output = json.dumps(str(prediction))            
    
    else:
        raise ValueError(f"Format {accept} is not supported. Please use on of {[JSON_CONTENT_TYPE, CSV_CONTENT_TYPE, PICKLE_CONTENT_TYPE]}")
    
    logging.info("Output serilaization sucessfully completed. ")
    
    # returning the serilaization value to client.
    return output