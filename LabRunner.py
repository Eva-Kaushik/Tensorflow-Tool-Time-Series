'''
Created on March 30, 2019

@author: JJ Sun
'''
from Utils import *

import keras
import numpy
import math

#Input data file in CSV format
RAW_INPUT_FILE_SINGLE_INPUT = 'THE PATH OF THE INPUT FILE'

def create_hyper_params_model_1():
    '''
    Example function to create a HyperParams object containing your hyperparameter configurations including
        * Type of your neural network, see RnnCellType for more details
        * Number of the model inputs
        * Number of time steps
        * Layer configurations, see LayerConfig for more details
        * Number of the model outputs
        * Loss function
        * Optimization algorithm
        * Learning rate
        * Batch size for mini-batch training
        * Number of traing epochs
        * Test window - The length of the data(not included in training) which will be used for predction at the end
                        of each epoch to generate a graph, the purpose of this is to generate 'snapshots' of the model
                        performance to help you visually check the progress of the training
    '''
    cell_type = RnnCellType.GRU_BIDI
    num_inputs = 2
    num_time_steps = 30
    loss='mean_squared_error'
    optimizer='adam'
    
    layer_config_1 = LayerConfig(100, 'tanh', kernel_initializer = 'glorot_uniform', dropout_rate = 0.3, is_bidi = True)
    layer_config_2 = LayerConfig(100, 'tanh', kernel_initializer = 'glorot_uniform', dropout_rate = 0.3, is_bidi = True)
    layer_config_3 = LayerConfig(100, 'tanh', kernel_initializer = 'glorot_uniform', dropout_rate = 0.3, is_bidi = True)
    layer_configs = [layer_config_1, layer_config_2, layer_config_3]
    num_outputs = 2
    learning_rate = 0.0001
    batch_size = 30
    test_window = 60
    epochs = 20
    initial_epoch = 0
    checkpoint_id = 0
    return HyperParams(RAW_INPUT_FILE_SINGLE_INPUT, 
                       cell_type, 
                       num_inputs, 
                       num_time_steps, 
                       num_outputs, 
                       learning_rate, 
                       layer_configs, 
                       batch_size, 
                       epochs, 
                       test_window, 
                       checkpoint_id,
                       initial_epoch,
                       loss, optimizer
                       )

# Templdate methods to define your other models
def create_hyper_params_model_2():
    pass
    
def create_hyper_params_model_3():
    pass
    
def get_current_hyperparemters():
    return create_hyper_params_model_1()

if __name__ == '__main__':
    
    # Initially, you will need to set these two flags to False to start experiment on a new model
    is_resume = False
    is_predicting = False

    init_logging()    
    
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    
    hparams = get_current_hyperparemters()
    
    runner = LabRunner(hparams)
    
    if is_resume:   
        # Get current model id
        hparams.model_id = get_current_model_id()
        log.info("In resume mode, loading existing model with id: %d", hparams.model_id)
        
        # Adjust total epochs and initial epochs in resume mode
        hparams.epochs = 50
        # Retrieve the latest checkpint then use it to populate initial_epoch and checkpoint_id
        hparams.initial_epoch = hparams.checkpoint_id = 999
        
        runner.load_and_resume_training()
    elif is_predicting:
        steps_to_predict = 10
        runner.load_and_predict(get_current_model_id(), steps_to_predict)
    else:    
        log.info("Not in resume mode, creating new model...")
        runner.train_new_model()
