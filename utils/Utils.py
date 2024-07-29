'''
Created on March 30, 2019

@author: JJ Sun
'''

import os

import logging
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras import regularizers
from keras.callbacks import CSVLogger, Callback, ModelCheckpoint, LambdaCallback
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta

from timeit import default_timer as timer
import keras
import numpy
import math
from keras.layers.wrappers import Bidirectional

logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
from numpy import NaN

import datetime 
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Adjust your root model location
MODLE_FOLDER = "C:/tensorlab_models_keras"
MODLE_ID_FILE = MODLE_FOLDER + "/model_id"
MODLE_SUB_FOLDER_SNAPSHOTS = "snapshots"
MODLE_SUB_FOLDER_CHECKPOINTS = "checkpoints"

INITIAL_MODLE_FILE_ID = 0
REPORTING_PERIOID = 100

LOG_FORMAT = '%(asctime)s %(levelname)s:%(message)s'
LOG_DATE_FORMAT = '%I:%M:%S'
IDX_COL_NAME = 'yymmdd'

log = None

class LabContext:
    def __init__(self):
        self.hparams = None
        self.lab_folder = None
        self.data_container = None
        self.model = None
        self.test_x = None
        self.test_y = None
        
lab_ctx = LabContext()
       
def init_logging():
    global log
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, datefmt=LOG_DATE_FORMAT)
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    
    tf.logging.set_verbosity(tf.logging.ERROR)

def init_model_id(value):
    '''
    Try to initialize a model ID file, return True if successful, which indicates a new model ID file
    has been created with value defined by constant INITIAL_MODLE_FILE_ID
    '''
    if not os.path.isfile(MODLE_ID_FILE):
        with open(MODLE_ID_FILE, 'w') as id_file:
            log.info("init model id file with value:%d", value)
            id_file.write('{}'.format(value))
            return True
    return False

def get_time_stamp():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%dT%H:%M:%S')

def increase_model_id():
    init_model_id(INITIAL_MODLE_FILE_ID)
    with open(MODLE_ID_FILE, 'r+') as id_file:
        model_id = int( (id_file.readline().split() )[0])
        model_id += 1
        id_file.seek(0,0)
        id_file.write('{}'.format(model_id))
    return model_id

def get_current_model_id():
    '''
    Return current model ID
    '''
    if init_model_id(INITIAL_MODLE_FILE_ID):
        return INITIAL_MODLE_FILE_ID
    else:
        with open(MODLE_ID_FILE, 'r') as id_file:
            model_id = int( (id_file.readline().split() )[0])
            return model_id

def get_model_paths(model_id):
    path_checkpoints = os.path.join(MODLE_FOLDER, str(model_id), MODLE_SUB_FOLDER_CHECKPOINTS)
    path_snapshots = os.path.join(MODLE_FOLDER, str(model_id), MODLE_SUB_FOLDER_SNAPSHOTS)
    
    return LabFolder(model_id, path_checkpoints, path_snapshots)

class LabFolder:
    def __init__(self, model_id, path_checkpoints, path_snapshots):
        self.model_id = model_id
        self.path_checkpoints = path_checkpoints
        self.path_snapshots = path_snapshots
        
def init_folder_structure():
    '''
    Initialize folders to store model checkpoints and summary data, all intermediate folders will be created
    
    Returns: 
        model_id: The newly assinged model id
        path_checkpoints: The path of the checkpoints folder for the new model
    '''
    # Create root 'models' folder at first
    if not os.path.isdir(MODLE_FOLDER):
        os.makedirs(MODLE_FOLDER)
        
    model_id = increase_model_id()
    lab_folder = get_model_paths(model_id)

    log.debug('Creating folder: %s', lab_folder.path_checkpoints)
    log.debug('Creating folder: %s', lab_folder.path_snapshots)
    
    if not os.path.isdir(lab_folder.path_checkpoints):
        os.makedirs(lab_folder.path_checkpoints)
    if not os.path.isdir(lab_folder.path_snapshots):
        os.makedirs(lab_folder.path_snapshots)
        
    return lab_folder

class HyperParams:
    """
    Construct a HyperParams object to hold training hyperparameters
    
    Args:
        input_file (str): The path of your input file, in csv format
        cell_type (RnnCellType): The type of the cell to be used for the training
        num_inputs (int): The number of inputs/features at each step
        num_time_steps (int): The number of steps for each test sequence
        num_outputs (int):
        learning_rate (int):
        layer_configs (list):
        batch_size (int):
        epochs (int):
        test_window (int):
        checkpoint_id (int):
        initial_epoch (int):
        loss (str):
        optimizer (str):
        kernel_initializer (keras.initializers):
    """
    def __init__(self, input_file, cell_type, num_inputs, num_time_steps, num_outputs, learning_rate, 
                 layer_configs, batch_size, epochs, test_window, 
                 checkpoint_id = -1,
                 initial_epoch = 0,
                 loss='mean_squared_error', 
                 optimizer='adam',
                 kernel_initializer = None):

        self.model_id = -1
        self.cell_type = cell_type
        self.num_inputs = num_inputs
        self.num_time_steps = num_time_steps
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.layer_configs = layer_configs
        self.kernel_initializer = kernel_initializer
        self.input_file = input_file
        self.test_window = test_window
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.initial_epoch = initial_epoch
        self.checkpoint_id = checkpoint_id
        
    def to_string(self):
        time_stamp = get_time_stamp()
        layer_configs_string = build_layer_config_string(self.layer_configs)
        string = """ModelId:{}, CheckPoint:{}, CellType:{}, NumInput:{},NumTimeSteps:{},LayerConfig:(\n{}),\nLearningRate:{},
                    BatchSize:{}, LossFunction:{}, Optimizer:{}, Timestamp:{}""".\
            format(self.model_id, self.checkpoint_id, self.cell_type, self.num_inputs, 
                   self.num_time_steps, layer_configs_string, self.learning_rate, 
                   self.batch_size,
                   self.loss, self.optimizer,
                   time_stamp)
        return string
    
    def is_bidi(self):
        return self.cell_type in [RnnCellType.GRU_BIDI, RnnCellType.LSTM_BIDI]
    
class RnnCellType(Enum):
    BasicRNN = 1
    RNN = 2
    GRU = 3
    LSTM = 4
    GRU_BIDI = 5
    LSTM_BIDI = 6

class LayerConfig:
    def __init__(self, num_units, activation, kernel_initializer = None, regularizition_l2 = 0, dropout_rate = 0, is_bidi = False):
        self.num_units = num_units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.is_bidi = is_bidi
        self.regularizition_l2 = regularizition_l2
        self.dropout_rate = dropout_rate
    
    def to_string(self):
        if self.kernel_initializer is None:
            kernel_initializer_str = 'Default'
        else:
            kernel_initializer_str = self.kernel_initializer
        return "NumUnits:{}, Activation:{}, KernalInitializer:{}, Dropout:{}, L2:{}".format(self.num_units, self.activation, 
                kernel_initializer_str, self.dropout_rate, self.regularizition_l2)
        
def build_layer_config_string(layer_configs):
    layer_configs_string = ''
    idx = 1
    for layer in layer_configs:
        layer_config = "Layer[{}]:[{}],\n".format(idx, layer.to_string()) 
        layer_configs_string += layer_config
        idx+=1
    
    return layer_configs_string.strip(',\n')

class PredictionDataContainer:
    def __init__(self, samples, scaler, prediction_seed_scaled, prediction_window):
        self.samples = samples
        self.scaler = scaler
        self.prediction_seed_scaled = prediction_seed_scaled
        self.prediction_window = prediction_window
        
    def build_prediction(self, predicted, num_inputs):
        last_existing_day = self.samples.index[-1]
        first_predicted_day = last_existing_day + datetime.timedelta(days=1)
        # Create new index for the predicted rows
        predicted_dates = pd.date_range(first_predicted_day, periods=self.prediction_window)
        
        # Create two copies, one has the new 'Predicted' column for plotting purse, another one
        # holds the predicted data in the same column so it can be saved for reporting 
        copy_for_plotting = self.samples.copy()
        
        # Add new columns to hold the predicted data, so when we are plotting, different
        # colors can be used for historical data and predicted data
        if num_inputs == 1:
            copy_for_plotting['Predicted'] = NaN
            df_predicted_for_plotting = pd.DataFrame(predicted, index=predicted_dates, columns=['Predicted'])
        else:
            new_cols_for_predicted = []
            for i in range(num_inputs):
                new_column_name = 'p_{}'.format(i)
                new_cols_for_predicted.append(new_column_name)
                copy_for_plotting[new_column_name] = NaN
                
            df_predicted_for_plotting = pd.DataFrame(predicted, index=predicted_dates, columns=new_cols_for_predicted)
        
        return (df_predicted_for_plotting, copy_for_plotting.append(df_predicted_for_plotting, ignore_index=False))
        
        
def build_prediction_data_container(hparams, prediction_window):
    samples = pd.read_csv(hparams.input_file, index_col=IDX_COL_NAME)
    samples.index = pd.to_datetime(samples.index)
    
    prediction_seed = samples.tail(hparams.num_time_steps)

    scaler = MinMaxScaler()
    prediction_seed_scaled = scaler.fit_transform(prediction_seed)
    
    log.info("prediction_seed_scaled shape: %s", str(prediction_seed_scaled.shape))
    
    dataContainer = PredictionDataContainer(samples, scaler, prediction_seed_scaled, prediction_window)
    return dataContainer

def get_model_file_paths(hparams):
    model_file = os.path.join(MODLE_FOLDER, str(hparams.model_id), "model-{}.json".format(hparams.model_id))
    
    check_point_id = str(hparams.checkpoint_id).zfill(2)
    
    weights_file = os.path.join(MODLE_FOLDER, 
                                str(hparams.model_id), 
                                MODLE_SUB_FOLDER_CHECKPOINTS,
                                "checkpoint-{}.hdf5".format(check_point_id))
    log.info("get_model_file_paths model_file: %s", model_file)
    log.info("get_model_file_paths weights_file: %s", weights_file)
    return model_file, weights_file

def persist_model(model, model_file):
    # serialize model to JSON
    log.info("Serializing model to: %s", model_file)
    model_json = model.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)
    log.info("Serializing model done") 
    
def persist_weights(model, weights_file):
    # serialize weights to HDF5
    log.info("Serializing weights to: %s", weights_file)
    model.save_weights(weights_file)
    log.info("Serializing weights done")
            
def persist_model_and_weights(model, hparams):
    model_file, weights_file = get_model_file_paths(hparams)
    persist_model(model, model_file)
    persist_weights(model, weights_file)
     
def load_model(hparams):
    model_file, weights_file = get_model_file_paths(hparams)
    log.info("Loading model from file: %s", model_file)
    
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    loaded_model = model_from_json(loaded_model_json)
    log.info("Model loaded, loading weights from file: %s", weights_file)
    
    # load weights into new model
    loaded_model.load_weights(weights_file)
    
    log.info("Weights loaded, compiling model")
    
    optimizer = create_optimizer(hparams)
    loaded_model.compile(loss=hparams.loss, optimizer=optimizer, metrics=[hparams.loss])

    log.info("Returning model")
    
    return loaded_model   

def update_and_save(hparams, df_predicted, path_snapshots):
    existing_records = pd.read_csv(hparams.input_file, index_col=IDX_COL_NAME)
    # Note that each row of the DataFrame is a Series, and the name of it is the value of
    # index of row
    for idx in df_predicted.index:
        idx_string = idx.strftime('%Y-%m-%d')
        
        if hparams.num_inputs == 1:
            # TODO Use the actual column name in your input file!
            row = pd.Series({'<COLUMN_NAME>': str(df_predicted.loc[idx]['Predicted'])}, name=idx_string)
        else:
            predicted_value = {}
            for i in range(hparams.num_inputs):
                new_column_name = 'p_{}'.format(i)
                predicted_value[new_column_name] = str(df_predicted.loc[idx][new_column_name])
                row = pd.Series(predicted_value, name=idx_string) 
                
        existing_records = existing_records.append(row)
    
    file_name = 'prediction-checkpoint-{}.csv'.format(hparams.checkpoint_id)
    csv_path = os.path.join(path_snapshots, file_name)
    log.info("Saving file: %s", csv_path)
    existing_records.to_csv(csv_path)
    
class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """

    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "Epoch: %i, %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)

class LabRunner:
    def __init__(self, hparams):
        self.data_container = None
        self.path_snapshots = None
        
        self.hparams = hparams
        
        init_logging()    
    
    def train_new_model(self):
        lab_folder = init_folder_structure()
        
        self.hparams.model_id = lab_folder.model_id 
        model = create_keras_model(self.hparams)
        
        init_file_logging(self.hparams.model_id)
        
        self.path_checkpoints = lab_folder.path_checkpoints
        self.path_snapshots = lab_folder.path_snapshots
        
        lab_ctx.hparams = self.hparams
        lab_ctx.lab_folder = lab_folder
        lab_ctx.model = model
        
        model_file, _ = get_model_file_paths(self.hparams)
        persist_model(model, model_file)
        
        self.train_model(model, self.hparams)

    def load_and_resume_training(self):
        # Get current model id
        self.hparams.model_id = get_current_model_id()
        
        init_file_logging(self.hparams.model_id)
         
        log.info("In resume mode, loading existing model with id: %d", self.hparams.model_id)
        
        # Reconstruct the model folder paths based on model id
        lab_folder = get_model_paths(self.hparams.model_id)
        
        self.path_checkpoints = lab_folder.path_checkpoints
        self.path_snapshots = lab_folder.path_snapshots
        
        model = load_model(self.hparams)
        
        lab_ctx.hparams = self.hparams
        lab_ctx.lab_folder = lab_folder
        lab_ctx.model = model
        
        self.train_model(model, self.hparams)
    
    @staticmethod
    def take_snap_shot(epoch, logs):
        log.info("take_snap_shot - Start, epoch: %d", epoch)
        
        testPredict = lab_ctx.model.predict(lab_ctx.test_x)
        testPredict = lab_ctx.data_container.scaler.inverse_transform(testPredict)
        
        testY = lab_ctx.data_container.scaler.inverse_transform(lab_ctx.test_y)
        testPredict = testPredict.astype(int)
        testY = testY.astype(int)
        
        log.debug("testPredict shape: %s", testPredict.shape)
        log.debug("test_set shape: %s", lab_ctx.data_container.test_set.shape)
        
        if lab_ctx.hparams.num_inputs > 1:
            for i in range(lab_ctx.hparams.num_inputs):
                new_column_name = 'p_{}'.format(i)
                lab_ctx.data_container.test_set.loc[:, new_column_name] = testPredict[:, i]
        else:
            lab_ctx.data_container.test_set.loc[:, 'Predicted'] = testPredict
            
        plotted = lab_ctx.data_container.test_set.plot(title=lab_ctx.hparams.to_string())
        figure = plotted.get_figure()
        figure.set_size_inches(20, 15)
        image_path = os.path.join(lab_ctx.lab_folder.path_snapshots, str(lab_ctx.hparams.model_id))
        
        figure.savefig(image_path + '.' + str(epoch) + '.png')
        
        # Close the figure otherwise it will be left in memory
        plt.close(figure)         
        log.info("take_snap_shot - Done, epoch: %d", epoch)
    
    def train_model(self, model, hparams):
        log.info("train_model - Start")
        
        overall_time_start = timer()
        
        model.summary() 
        
        # Note this doesn't work with my GPU, feel free to try!
#         tensorboard_cb = keras.callbacks.TensorBoard(
#             log_dir=self.path_summary, 
#             histogram_freq=1,
#             write_graph=1,
#             write_grads=1,
#             update_freq = 365,
#             write_images=False)

        logging_cb = LoggingCallback(log.info)
        checkpoint_cb = ModelCheckpoint(filepath=os.path.join(self.path_checkpoints, 'checkpoint-{epoch:02d}.hdf5'))
    
        data_container = build_train_data_container(hparams)
        trainX, trainY = build_sequential_training_data_set(data_container.train_scaled, look_back_window=hparams.num_time_steps)
        log.debug("trainX shape: %s", trainX.shape)
        log.debug("trainY shape: %s", trainY.shape)
        
        testX, testY = build_sequential_test_data_set(data_container.samples_scaled, hparams.test_window, look_back_window=hparams.num_time_steps)
        
        lab_ctx.data_container = data_container
        lab_ctx.test_x = testX
        lab_ctx.test_y = testY
        
        # The LSTM network expects the input data (X) to be provided with a specific array 
        # structure in the form of: [samples, time steps, features]     
        trainX = numpy.reshape(trainX, (trainX.shape[0], hparams.num_time_steps, hparams.num_inputs))
        
        #TODO - Use previous file if we are in resume mode
        model_fit_log = create_model_fit_log_file_full_name(hparams.model_id)
        csv_logger = CSVLogger(model_fit_log, append=True, separator=',')
        
        predict_cb = LambdaCallback(on_epoch_end=LabRunner.take_snap_shot)
        
        model.fit(trainX, trainY, 
                epochs = hparams.epochs, 
                batch_size = hparams.batch_size, 
                verbose=1,
                validation_split=0.09, 
                initial_epoch = hparams.initial_epoch,
                shuffle = True,
#                 Callbacks that work with GPU
                callbacks = [csv_logger, logging_cb, checkpoint_cb, predict_cb]
#                 Try tensorboard_cb if you want
#                 callbacks = [csv_logger, logging_cb, checkpoint_cb, tensorboard_cb, predict_cb]
                  )
        
        
        log.info("Training finished.")
    
        hparams.checkpoint_id += (hparams.epochs - hparams.initial_epoch)
        
        log.info("Checkpoint id has been updated to: %d", hparams.checkpoint_id)
        
        model = load_model(hparams)
        
        log.info("Model reloaded.")
        
        model.summary()
        
        log.info("Predicting...")
        
        testPredict = model.predict(testX)
        
        testPredict = data_container.scaler.inverse_transform(testPredict)
        testY = data_container.scaler.inverse_transform(testY)
        testPredict = testPredict.astype(int)
        testY = testY.astype(int)
        
        log.debug("testPredict shape: %s", testPredict.shape)
        log.debug("test_set shape: %s", data_container.test_set.shape)
        
        if hparams.num_inputs > 1:
            for i in range(lab_ctx.hparams.num_inputs):
                new_column_name = 'p_{}'.format(i)
                data_container.test_set.loc[:, new_column_name] = testPredict[:, i]
        else:
            data_container.test_set.loc[:, 'Predicted'] = testPredict
            
        plotted = data_container.test_set.plot(title=hparams.to_string())
        figure = plotted.get_figure()
        figure.set_size_inches(20, 15)
        image_path = os.path.join(self.path_snapshots, str(hparams.model_id))
        
        #TODO - Don't overwrite old image
        figure.savefig(image_path + '.final.png')
        
        # Close the figure otherwise it will be left in memory
        plt.close(figure)
        
        overall_time_end = timer()

        log.info("Total time taken: %f minutes", (overall_time_end-overall_time_start)/60)    

        log.info("train_model - Done")
        
    def load_and_predict(self, model_id, prediction_window):
        log.info("load_and_predict - Start")
        
        init_file_logging(model_id)
        self.hparams.model_id = model_id
        # Reconstruct the model folder paths based on model id
        lab_folder = get_model_paths(self.hparams.model_id)
        data_container = build_prediction_data_container(self.hparams, prediction_window)
        model = load_model(self.hparams)
        model.summary()
        
        input_steps = self.hparams.num_time_steps
        num_inputs = self.hparams.num_inputs
    
        '''
        Take the last block of the prediction seed as the training seed, then start 
        moving forward step by step to build the predicted window
        
        Note the prediction_seed_scaled is a scaled ndarray with shape (#samples, #features)
        '''
        train_seed = list(data_container.prediction_seed_scaled[-input_steps:])
        log.debug("predict - data_container.prediction_seed_scaled shape: %s", data_container.prediction_seed_scaled.shape)
        log.debug("predict - train_seed type: %s", type(train_seed))
        log.debug("predict - data_container.prediction_seed_scaled[-input_steps:] type: %s", type(data_container.prediction_seed_scaled[-input_steps:]))
        log.debug("predict - data_container.prediction_seed_scaled[-input_steps:] shape: %s", data_container.prediction_seed_scaled[-input_steps:].shape)
    
        for i in range(prediction_window):
            indd = np.array(train_seed[-input_steps:])
            
            input_block = indd.reshape(-1, input_steps, num_inputs)
            # The output should only be one predicted step in shape (1, #features)
            y_pred = model.predict(input_block)
            
            log.debug("load_and_predict - y_pred[%d/%d]: %s", i, prediction_window, str(y_pred))
            
            train_seed.append(y_pred[0])
        
        # Note actual_result is a list
        actual_result = data_container.scaler.inverse_transform( np.array( train_seed[-prediction_window:] ).reshape(prediction_window, num_inputs) )
        actual_result = actual_result.astype(int)

        # Build the predicted 'picture' of the future
        df_predicted, full_data_with_prediction = data_container.build_prediction(actual_result, num_inputs)

        # Export to csv
        update_and_save(self.hparams, df_predicted, lab_folder.path_snapshots)

        first_existing_day = data_container.samples.index[0]
        last_existing_day = data_container.samples.index[-1]
        first_predicted_day = df_predicted.index[0]
        last_predicted_day = df_predicted.index[-1]
        
        log.info("Prediction - First existing day is: %s", str(first_existing_day))
        log.info("Prediction - last existing day is: %s", str(last_existing_day))
        log.info("Prediction - First predicted day is: %s", str(first_predicted_day))
        log.info("Prediction - last predicted day is: %s", str(last_predicted_day))
        
        assert len(full_data_with_prediction) == prediction_window + len(data_container.samples)
        
        plotted = full_data_with_prediction.plot(title=self.hparams.to_string())
        
        # Save the plot
        figure = plotted.get_figure()
        figure.set_size_inches(20, 15)
        part_file_name = 'prediction-checkpoint-{}'.format(self.hparams.checkpoint_id)
        image_path = os.path.join(lab_folder.path_snapshots, part_file_name)
        
        log.info("Saving image: %s", image_path)
        figure.savefig(image_path + '.png')

def create_rnn_instance(hparams, layer, return_sequences):
    rnn = None
    log.info("Create %s layer with config: %s", str(hparams.cell_type), layer.to_string())
    
    if hparams.cell_type == RnnCellType.LSTM_BIDI:
        rnn = Bidirectional(
                LSTM(
                        layer.num_units, 
                        activation = layer.activation,
                        kernel_initializer = layer.kernel_initializer,
                        kernel_regularizer = regularizers.l2(layer.regularizition_l2) if (layer.regularizition_l2 > 0) else None,
                        return_sequences = return_sequences), 
                merge_mode='concat',
                # The shape of the input block
                input_shape=(hparams.num_time_steps, hparams.num_inputs))
    elif hparams.cell_type == RnnCellType.GRU_BIDI:
        rnn = Bidirectional(
                GRU(
                        layer.num_units, 
                        activation = layer.activation,
                        kernel_initializer = layer.kernel_initializer,
                        kernel_regularizer = regularizers.l2(layer.regularizition_l2) if (layer.regularizition_l2 > 0) else None,
                        return_sequences = return_sequences), 
                merge_mode='concat',
                # The shape of the input block
                input_shape=(hparams.num_time_steps, hparams.num_inputs))
    return rnn
    
def create_keras_model(hparams):
    log.info("create_keras_model - Start")
    
    model = Sequential()
    layers = hparams.layer_configs
    layer = hparams.layer_configs[0];
    
    # Expected input shape: #inputs, input sequence length, length of each input vector
    
    if hparams.is_bidi:
        for layer in layers:
            index = layers.index(layer)
            # return output sequence if this is not the last recurrent layer
            return_sequences = (index != len(layers)-1)
            
            if index == 0:
                # The first layer needs to provide the shape of the input
                log.info("Creating Bidirectional Layer - %d, %s", index, layer.to_string())
                log.info("Return output sequence: %s", return_sequences)
                
                rnn = create_rnn_instance(hparams, layer, return_sequences)
                model.add(rnn)
                
            else:
                log.info("Creating Bidirectional Layer - %d, %s", index, layer.to_string())
                log.info("Return output sequence: %s", return_sequences)
                
                rnn = create_rnn_instance(hparams, layer, return_sequences)
                model.add(rnn)
                
            # Add dropout layer to hidder layer
            if layer.dropout_rate > 0:
                log.info("Adding Dropout layer with rate: %f", layer.dropout_rate)
                model.add(Dropout(layer.dropout_rate))
    else:
        """
            Add a LSTM layer into the model, note that the 'input_shape' is used to describe the shape of ONE input block
            whose shape should be (length of the sequence, number of features of each item in the sequence), if you want to 
            enable batch training, use another parameter 'batch_input_shape' 
        """
        model.add(LSTM(layer.num_units, kernel_regularizer = regularizers.l2(0.01), input_shape=(hparams.num_time_steps, hparams.num_inputs), return_sequences=False))
        
    """
        Add a Dense layer to project the output from previous LSTM layer to required shape, note that no activation function 
        is used for the output layer because it is a regression problem and we are interested in predicting numerical 
        values directly without transform
    """
    
    # Create optimizer
    optimizer = create_optimizer(hparams)
    
    # Add Dense layer as the last layer to project hidden output to final output
    model.add(Dense(hparams.num_outputs))
    
    
    model.compile(loss=hparams.loss, optimizer=optimizer, metrics=[hparams.loss])

    log.info("create_keras_model - Done")
    return model

def create_optimizer(hparams):
    op = None
    log.info("Creating {} optimizer with learning rate {}".format(hparams.optimizer, hparams.learning_rate))
    if hparams.optimizer == 'adam':
        op = Adam(lr=hparams.learning_rate)
    elif hparams.optimizer == 'sgd':
        op = SGD(lr=hparams.learning_rate, nesterov=True)
    elif hparams.optimizer == 'adagrad':
        op = Adagrad(lr=hparams.learning_rate)
    elif hparams.optimizer == 'rmsprop':
        op = RMSprop(lr=hparams.learning_rate)   
    elif hparams.optimizer == 'adadelta':
        op = Adadelta(lr=hparams.learning_rate)      
    return op

def create_log_file_full_name(model_id):
    timestamp = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
    model_folder = os.path.join(MODLE_FOLDER, str(model_id))
    log_file_full_name = os.path.join(model_folder, 'training_' + str(model_id) + '_' + timestamp+'.log')
    log.info('Created full log file name: %s', log_file_full_name)

    return log_file_full_name


def create_model_fit_log_file_full_name(model_id):
    # Note that we want to use a single log file, if the training is resumed, new log entries will be
    # appended to the same file
    model_folder = os.path.join(MODLE_FOLDER, str(model_id))
    log_file_full_name = os.path.join(model_folder, 'model_fit_' + str(model_id) + '.csv')
    log.info('create_model_fit_log_file_full_name: %s', log_file_full_name)

    return log_file_full_name


def init_file_logging(model_id):
    file_name = create_log_file_full_name(model_id)
    log_file_handler = logging.FileHandler(file_name)
    log_file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    log_file_handler.setFormatter(formatter)

    log.addHandler(log_file_handler)
    
class TrainDataContainer:
    def __init__(self, samples, train_set, test_set, scaler, samples_scaled, train_scaled, test_scaled):
        self.samples = samples
        self.train_set = train_set
        self.test_set = test_set
        self.scaler = scaler
        self.samples_scaled = samples_scaled
        self.train_scaled = train_scaled
        self.test_scaled = test_scaled

def build_train_data_container(hparams):
    """
    Loads the sample data, then splits it into train and test sets, the train set is also scaled
    
    Args:
        hparams: 
            The HyperParam object containing hyperparameters
        test_window_length:
            The length of the test set that will be split from the raw sample data set
        
    Returns:
        dataContainer: 
            a TrainDataContainer object containing the raw sample data, training set, test set, 
            scaled traing set and the scaler
    """
    
    log.info('build_train_data_container - start')
    
    samples = pd.read_csv(hparams.input_file, index_col='yymmdd')
    samples.index = pd.to_datetime(samples.index)
    rows, _ = samples.shape
    
    log.info('build_train_data_container - Raw sample data type: %s, shape: %s', type(samples), samples.shape)
    
    # Split to train and test set
    train_set = samples.head(rows - hparams.test_window)
    test_set = samples.tail(hparams.test_window)
    
    log.debug('build_train_data_container - train_set type: %s, shape: %s', type(train_set), train_set.shape)
    log.debug('build_train_data_container - test_set type: %s, shape: %s', type(test_set), test_set.shape)

    # Scaler/Normalize, note the default range is (0, 1), note the output is an ndarray
    scaler = MinMaxScaler(feature_range=(0, 1))
    samples_scaled = scaler.fit_transform(samples)
    
    # Split to train set and test set
    train_scaled = samples_scaled[0:rows - hparams.test_window, :]
    test_scaled = samples_scaled[-hparams.test_window:, :]
    
    # Expected shape should be: [#samples, #features]
    log.info("build_train_data_container - training set normalized with type: %s, shape: %s", type(train_scaled), train_scaled.shape)
    
    dataContainer = TrainDataContainer(samples, train_set, test_set, scaler, samples_scaled, train_scaled, test_scaled)
    
    log.info('build_train_data_container - done')
    return dataContainer

def next_batch(training_data, hparams):
    steps = hparams.num_time_steps
    num_inputs = hparams.num_inputs
    random_start = np.random.randint(0, len(training_data)-steps)
    selected_window = np.array(training_data[random_start:random_start + steps + 1]).reshape(num_inputs, steps+1)
    # reshape input to be [samples, time steps, features]
    train_batch = selected_window[:, :-1].reshape(-1, steps, num_inputs)
    test_batch = selected_window[:, 1:].reshape(-1, steps, num_inputs)
    return train_batch, test_batch

def build_sequential_training_data_set(training_data, look_back_window=1):
    """
    Builds test data sets(i.e., inputs and expected outputs) from the training data set

    Args:
        training_data: An ndarray that contains the training data
        look_back_window: An integer value indicating the length of the look-back window 
        
    Returns:
        dataX: A list containing the ndarrays, each item(i.e., an array) in the list 
               represents the sequential input data with the specified window length
        dataY: A list containing the ndarrays, each item(i.e., an array) in the list represents 
               the corresponding output vector
        
    """
    # Expected shape of the training_data should be: [#samples, #features]
    
    log.debug("build_sequential_training_data_set - start")
    log.debug("build_sequential_training_data_set - training_data type: %s, shape: %s, look_back_window: %d", 
              type(training_data), training_data.shape, look_back_window)
     
    # Each item in the list is a window of sequence data
    dataX, dataY = [], []
    for i in range(len(training_data)-look_back_window-1):
        training_input = training_data[i:(i+look_back_window), :]
        # Shift by one record to get the expected output of the prediction
        training_output = training_data[i + look_back_window, :]
        dataX.append(training_input)
        dataY.append(training_output)
        
    log.debug("build_sequential_training_data_set - dataX len: %s", len(dataX))
    log.debug("build_sequential_training_data_set - dataY len: %s", len(dataY))
    
    assert len(dataX) == len(dataY)
    
    # Ensure the output is corret - The value of dataY at position k-1 should be 
    # equal to last value of dataX at position k
    for i in range(1, len(dataX)):
        assert (dataX[i][-1] == dataY[i-1]).all()
    
    log.debug("build_sequential_training_data_set - done")
    
    return np.array(dataX), np.array(dataY)
    
def build_sequential_test_data_set(seed_data, test_window, look_back_window=1):
    """
    Builds sequential data set for testing.
        
    Args:
        seed_data: The ndarray seed data set from which the test data will be extracted
        test_window: The length of the data(i.e., the number of time steps) that will be predicted
        look_back_window: The length of the data/steps that we will use to predict the value ahead of the window
    """
    
    # Expected shape of the seed_data should be: [#samples, #features]
    log.debug("build_sequential_test_data_set - start")
    log.debug("build_sequential_test_data_set - seed_data type: %s, shape: %s, test_window: %d, look_back_window: %d", 
              type(seed_data), seed_data.shape, test_window, look_back_window)
    
    seed_base = seed_data[-(test_window + look_back_window):, :]
    
    # Each item in the list is a window of sequence data
    dataX, dataY = [], []
    for i in range( test_window ):
        training_input = seed_base[i:(i+look_back_window), :]
        # Shift by one record to get the expected output of the prediction
        training_output = seed_base[i + look_back_window, :]
        dataX.append(training_input)
        dataY.append(training_output)
        
    log.debug("build_sequential_test_data_set - dataX len: %s", len(dataX))
    log.debug("build_sequential_test_data_set - dataY len: %s", len(dataY))
    
    assert len(dataX) == len(dataY)
    
    # Ensure the output is correct - The value of dataY at position k-1 should be 
    # equal to last value of dataX at position k
    for i in range(1, len(dataX)):
        assert (dataX[i][-1] == dataY[i-1]).all()
    
    log.debug("build_sequential_training_data_set - done")
    
    return np.array(dataX), np.array(dataY)
