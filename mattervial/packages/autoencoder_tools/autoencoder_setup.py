from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU, Conv1D, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization, Conv1DTranspose, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import datetime

from keras.models import load_model
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.spatial.distance import correlation, cosine
import pickle
from matplotlib import pyplot
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
from typing import List

# Disable eager execution because of symbolic type problems 
from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

# Set the random seed in TensorFlow
tf.random.set_seed(42)


def get_encoder_decoder(model, 
                        central_layer_name,
                        mode='default'):
    layer = model.get_layer(central_layer_name)
    # encoder_input = Input(model.input_shape[1:] , name='input_input') # to avoid same name error
    encoder_input = Input(model.input_shape[-1] , name='input_input') # to avoid same name error
    #encoder_input = Input(shape=model.n_inputs[1:])
    # decoder_input = Input(layer.output_shape[1:])
    decoder_input = Input(layer.output_shape[-1])

    #decoder_input = Input(shape=layer.output_shape[1:])
    
    # encoder/decoder starts with the input layer
    encoder_output = encoder_input
    decoder_output = decoder_input
    # encoder/decoder goes through the layers appending each one.
    encoder = True
    for layer in model.layers:
        if encoder:
            # ## special case for VAE
            # if layer.name == 'bottleneck_zmean':
            #     z_mean = model.get_layer('bottleneck_zmean')
            #     z_log_var = model.get_layer('bottleneck_zlog')
            #     vae_bottleneck = Lambda(sampling,name='bottleneck')([z_mean, z_log_var])
            #     encoder_output = encoder_output(vae_bottleneck)
            # elif layer.name == 'bottleneck_zlog':
            #     pass
            # elif layer.name == 'bottleneck':
            #     encoder = False
            # else:
            encoder_output = layer(encoder_output)
        else:
            decoder_output = layer(decoder_output)
        if layer.name == central_layer_name:
            encoder = False

    encoder_model = Model(encoder_input, encoder_output)
    decoder_model = Model(decoder_input, decoder_output)

    return encoder_model, decoder_model

# # VAE sampling function
# def sampling(args):
#     z_mean, z_log_var = args

#     batch = tf.shape(z_mean)[0]
#     dim = tf.shape(z_mean)[1]
#     epsilon = tf.random.normal(shape=(batch, dim))
#     random_sample = z_mean + tf.exp(0.5 * z_log_var) * epsilon
#     return random_sample

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf

class SamplingLayer(Layer):
    def __init__(self, name='sampling', **kwargs):
        super(SamplingLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        random_sample = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return random_sample

class VAELossLayer(Layer):
    def __init__(self, loss, kl_loss_coeff=1.0, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.loss = loss
        self.kl_loss_coeff = kl_loss_coeff

    def call(self, inputs):
        x, x_decoded_mean, z_mean, z_log_var = inputs
        if self.loss == 'binary_crossentropy':
            xent_loss = tf.keras.losses.binary_crossentropy(x, x_decoded_mean)
        elif self.loss == 'mean_squared_error' or self.loss == 'mse':
            xent_loss = tf.keras.losses.mean_squared_error(x, x_decoded_mean)
        elif self.loss == 'huber' or self.loss == 'logcosh':
            xent_loss = tf.reduce_mean(tf.math.log(tf.math.cosh(x - x_decoded_mean)))
        else:
            raise ValueError('Unsupported loss function for VAE')
        
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = tf.keras.backend.mean(xent_loss + self.kl_loss_coeff * kl_loss)
        self.add_loss(vae_loss)
        return x_decoded_mean

    def get_config(self):
        config = super(VAELossLayer, self).get_config()
        config.update({'loss': self.loss, 'kl_loss_coeff': self.kl_loss_coeff})
        return config

class AsymmetricVAELossLayer(Layer):
    def __init__(self, loss, asymmetric_loss_weights, kl_loss_coeff=1.0, **kwargs):
        super(AsymmetricVAELossLayer, self).__init__(**kwargs)
        self.loss = loss
        self.kl_loss_coeff = kl_loss_coeff
        self.asymmetric_loss_weights = tf.convert_to_tensor(asymmetric_loss_weights)
        self.asymmetric_loss_weights = tf.reshape(tf.cast(self.asymmetric_loss_weights, tf.float32), (1, -1))
        self.asymmetric_loss_weights = tf.transpose(self.asymmetric_loss_weights)
        
    def call(self, inputs):
        x, x_decoded_mean, z_mean, z_log_var = inputs
        if self.loss == 'mean_squared_error' or self.loss == 'mse':
            xent_loss = tf.keras.losses.mean_squared_error(x, x_decoded_mean) 
            squared_diff = tf.keras.backend.square(x - x_decoded_mean) 
            assym_loss = tf.keras.backend.dot(squared_diff, self.asymmetric_loss_weights)
        else:
            raise ValueError('Unsupported loss function for VAE')
        
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = tf.keras.backend.mean(xent_loss + assym_loss + self.kl_loss_coeff * kl_loss)
        self.add_loss(vae_loss)
        return x_decoded_mean

    def get_config(self):
        config = super(AsymmetricVAELossLayer, self).get_config()
        config.update({'loss': self.loss, 'kl_loss_coeff': self.kl_loss_coeff,
                       'asymmetric_loss_weights': self.asymmetric_loss_weights})
        return config
def get_results_model(model,Xtest):
    y=Xtest
    ypred=model.predict(Xtest)
    resultsmodel=[]
    for metric in [correlation, cosine]:
        metric_result=np.array([metric(y[i],ypred[i]) for i in range(len(y))]).mean()
        print(metric.__name__,metric_result)
        resultsmodel.append(metric_result)
    results_mae=mean_absolute_error(y, ypred)
    resultsmodel.append(results_mae)
    print('MAE:',results_mae)
    rmse_result=np.sqrt(mean_squared_error(y, ypred))
    resultsmodel.append(rmse_result)
    print('RMSE:',rmse_result)
    r2_result=r2_score(y, ypred, multioutput='variance_weighted')
    resultsmodel.append(r2_result)
    print('r2:',r2_result)
    yzeros=np.zeros(y.shape)
    results_rmse0=np.sqrt(mean_squared_error(y, yzeros))
    resultsmodel.append(results_rmse0)
    print('RMSE zero-vector:',results_rmse0)
    resultsmodel = [ float(r) for r in resultsmodel]
    return resultsmodel

def hypertune_autoencoder(
        dataset : pd.DataFrame = None,
        prefix_name : str = None,
        bottleneck_ratios : List[float] = None, 
        batch_sizes : List[int] = None ,   # [16,32,64] 
        epochs_list : List[int] = None ,  # [50,100,200]
        loss_functions : List[str] = None , # ['mse','log_cosh']
        learning_rates : List[float] = None, # [0.0005, 0.001,0.005]
        architectures : List[str] = None, # 2n_b default, depends what is implemented.
        savedir : str = './',
        ):
## Log Hyperbolic Cosine Loss Improves Variational Auto-Encoder
## https://openreview.net/forum?id=rkglvsC9Ym
    # Loop through all combinations of hyperparameters
    for architecture in architectures:
      for bottleneck_ratio in bottleneck_ratios:
        for loss in loss_functions:
          for batch_size in batch_sizes:
            for epoch in epochs_list:
              for learning_rate in learning_rates:
                # Build and compile the autoencoder model with the current set of hyperparameters
                if savedir[-1] != '/':
                    savedir+='/'
                train_autoencoder( prefix_name = prefix_name,
                              dataset = dataset,
                              compress_ratio = bottleneck_ratio,
                              architecture=architecture,
                              batch_size=batch_size,
                              epochs = epoch,
                              loss = loss,
                              learning_rate = learning_rate,
                              savedir=savedir+f"{prefix_name}_{architecture['arch']}",
                             )
            
# def train_generator(batch_size):
#     # write your data generator here that yields batches of training data
#     # example: for x_train, y_train in your_data_generator(batch_size):
#     #            yield x_train, y_train
#     while True:
#         # generate batches of training data
#         x_train, y_train = your_data_generator(batch_size)
#         yield x_train, y_train

def train_autoencoder(prefix_name : str = 'Model', 
        dataset : pd.DataFrame = None,
        compress_ratio : float = 0.5,
        architecture : dict = {'arch':'default'},
        savedir : str = '',
        logfile : str = 'EncoderResults.txt', 
        epochs : int = 100,
        learning_rate : float = 0.001,
        batch_size : int = 16, 
        random_state : int = 1,
        loss : str = 'mse',
        return_results : bool = False,
        get_test_results : bool = True, ## False to avoid memory errors of tensorflow in genetic
        test_size_ratio : float = 0.1,
        #  show_summary : bool = True,
        **kwargs
        ):    
    ''' 
    Parameters
    ----------
    prefix_name : str, optional
        Name of the model, by default 'Model'
    dataset : pd.DataFrame, optional
        Data to train the model, by default None
    compress_ratio : float, optional
        Ratio of the compress layer, by default 0.5
    architecture : dict, optional
        Architecture of the model, by default {'arch':'default'}
    savedir : str, optional
        Directory to save the model, by default ''
    logfile : str, optional
        Logfile to save the results, by default 'EncoderResults.txt'
    epochs : int, optional
        Number of epochs, by default 100
    learning_rate : float, optional
        Learning rate, by default 0.001
    batch_size : int, optional
        Batch size, by default 16
    random_state : int, optional
        Random state, by default 1
    loss : str, optional
        Loss function to use, by default 'mse'
    return_results : bool, optional
        Return results, by default False
    test_size_ratio : float, optional
        Test size ratio, by default 0.1
    Returns
    -------
    model, results
        Returns the model and the results if return_results is True
    ''' 

    if 'n_factor' not in architecture.keys():
        architecture['n_factor'] = ''
    if prefix_name:
        logfile=f"{prefix_name}_{logfile}"
    name_encoder = f"{prefix_name}_{architecture['arch']}{architecture['n_factor']}_cr{compress_ratio}_bs{batch_size}_ep{epochs}_loss_{loss}_lr{learning_rate}"
    try:
        os.mkdir(savedir)
    except:
        print(f"{savedir} already created.")
    Xtoencode=dataset
    ## drop columns all 0 and register them
    columns_todrop = Xtoencode.columns[(Xtoencode == 0).all()]
    if savedir[-1] != '/':
        savedir+='/'
    with open(savedir+"dropped_columns.txt", 'w') as f:
        text=""
        for column in columns_todrop:
            text+=str(column)+"\n"
        f.write(text)
    Xtoencode = Xtoencode.loc[:, Xtoencode.any()]
    print(f"Shape of dataset to encode: {Xtoencode.shape}"  )
    ## save the columns that are encoded
    with open(savedir+"encoded_columns.txt", 'w') as f:
        text=""
        for column in Xtoencode.columns:
            text+=str(column)+"\n"
        f.write(text)
    # split into train test sets
    X_train, X_test, _, _ = train_test_split(Xtoencode, np.zeros(Xtoencode.shape[0]), 
                                             test_size=test_size_ratio, random_state=random_state)
    # save the id of the train and test sets in text files
    np.savetxt(savedir+"train_ids.txt", X_train.index.values, fmt='%s')
    np.savetxt(savedir+"test_ids.txt", X_test.index.values, fmt='%s')
    # scale data
    t = MinMaxScaler()
    t.fit(X_train)
    pickle.dump(t,open(savedir+f"Scaler_{prefix_name}.pkl","wb"))
    X_train = t.transform(X_train)
    X_test = t.transform(X_test)
    # number of input columns
    n_inputs = Xtoencode.shape[1]
    results_filename=savedir+logfile
    if not os.path.exists(results_filename):
        with open(results_filename, 'w') as f:
            f.write(f"# Training {prefix_name} # Initial Number of Features: {n_inputs}\n")
            entries=['architecture','loss_fn','batch_size','epochs','learning_rate',
                    'n_bottleneck_ratio','n_bottleneck', 'train_loss', 'val_loss', 'correlation',
                    'cosine dist', 'MAE', 'RMSE', 'R2', 'RMSE zero-vector']
            text=''
            for i in range(len(entries)):
                text+=f"{entries[i]:>18}|"
            text+='\n'
            f.write(text)
    
    results=[]
    if architecture['arch'] == 'default': 
        n_bottleneck=int(n_inputs*compress_ratio)
        model = create_autoencoder(n_inputs=n_inputs,
                                   layers_structure= [n_inputs*2, int(n_bottleneck) ],
                                   loss = loss, lr=learning_rate) 
    elif architecture['arch'] == '2n_m2nb_b':
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [n_inputs*2, int((n_inputs*2+n_bottleneck)/2), 
                           int(n_bottleneck) ]  
        model = create_autoencoder(n_inputs=n_inputs, layers_structure=layers_structure,
                                   loss = loss, lr=learning_rate) 
    elif architecture['arch'] == '2n_n_b':
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [n_inputs*2, n_inputs, int(n_bottleneck) ]  
        model = create_autoencoder(n_inputs=n_inputs, layers_structure=layers_structure,
                                   loss = loss, lr=learning_rate) 
    elif architecture['arch'] == 'n_2n_b':
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [n_inputs, 2*n_inputs, int(n_bottleneck) ]  
        model = create_autoencoder(n_inputs=n_inputs, layers_structure=layers_structure,
                                   loss = loss, lr=learning_rate) 
    elif architecture['arch'] == '3n_b':
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [3*n_inputs, int(n_bottleneck) ]  
        model = create_autoencoder(n_inputs=n_inputs, layers_structure=layers_structure,
                                   loss = loss, lr=learning_rate)
    elif architecture['arch'] == 'n_b':
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [n_inputs, int(n_bottleneck) ]  
        model = create_autoencoder(n_inputs=n_inputs, layers_structure=layers_structure,
                                   loss = loss, lr=learning_rate)
    elif architecture['arch'] == '2n_3n_b':
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [2*n_inputs, 3*n_inputs, int(n_bottleneck) ]  
        model = create_autoencoder(n_inputs=n_inputs, layers_structure=layers_structure,
                                   loss = loss )
    elif architecture['arch'] == '3n_2n_b':
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [3*n_inputs, 2*n_inputs, int(n_bottleneck) ]  
        model = create_autoencoder(n_inputs=n_inputs, layers_structure=layers_structure, 
                                   loss = loss, lr=learning_rate)
    elif architecture['arch'] == '2n_conv2_b':
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [2*n_inputs, int(n_bottleneck) ]  
        model = create_autoencoder(n_inputs=n_inputs, layers_structure=layers_structure, 
                                   loss = loss, lr=learning_rate, type_architecture = 'convoluted',
                                   conv_reduction = 2,
                                   )
    elif architecture['arch'] == 'custom_VAE':
         ## this is needed to use custom VAE because of symbolic operations
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [int(architecture['n_factor']*n_inputs), int(n_bottleneck) ]  
        # model, vae_loss = create_autoencoder(n_inputs=n_inputs, layers_structure=layers_structure, 
        #                            loss = loss, lr=learning_rate, type_architecture = 'VAE',
        #                            )
        if 'asymmetric_loss_weights' in architecture.keys():
            loss_weights = architecture['asymmetric_loss_weights'] 
            model = create_autoencoder(n_inputs=n_inputs, layers_structure=layers_structure, 
                                       loss = loss, lr=learning_rate, type_architecture = 'VAE',
                                       asymmetric_loss_weights = loss_weights)
        else:
            model = create_autoencoder(n_inputs=n_inputs, layers_structure=layers_structure, 
                                   loss = loss, lr=learning_rate, type_architecture = 'VAE',
                                   )              


    elif architecture['arch'] == 'custom_n_b':
        class TestFailed(Exception):
            pass
        if architecture['n_factor'] == '':
            raise TestFailed("you need to specify a n_factor in architecture to use custom_n_b.")
        n_bottleneck=int(n_inputs*compress_ratio)
        layers_structure= [architecture['n_factor']*n_inputs, int(n_bottleneck) ]  
        model = create_autoencoder(n_inputs=n_inputs, 
                                   layers_structure=layers_structure,
                                   loss = loss, lr=learning_rate)

    ## check if encoder already exists in folder, in this case goes to next 
    encoder_path=savedir+f'{name_encoder}_AutoEncoder.h5'
    LoadedModel=False
    if os.path.exists(encoder_path):
        print(f'File {name_encoder}_AutoEncoder.h5 exists in folder already, skiping this calculation.')
        if not return_results:
            return 0
        else: # load existing model to return results
            LoadedModel=True
            # model.get_layer('dense_1')
            model = load_model(encoder_path, custom_objects={'SamplingLayer': SamplingLayer, 'VAELossLayer':VAELossLayer},
                               compile=False)

    FailedTraining=False
    if not LoadedModel:
        try:
            # logs = "logs/" 

            # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
            #                                      histogram_freq = 1,
            #                                      profile_batch = '500,520')
            history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                                verbose=2, validation_data=(X_test,X_test),
                                # callbacks=[tboard_callback]
                                )
            # plot loss
            print(f"COMPRESSED VECTOR SIZE: {n_bottleneck}")    
            pyplot.plot(history.history['loss'], label='train')
            pyplot.plot(history.history['val_loss'], label='test')
            print(f"MSE loss in the autoencoder: {history.history['val_mse'][-1]}")
            mse_error_train = history.history['mse'][-1]
            mse_error_val = history.history['val_mse'][-1]
        except Exception as e:
            print(e)
            print("Training of this model failed.")
            mse_error_train='--'
            mse_error_val='--'
            FailedTraining=True
    else: # if model was loaded
        mse_error_train='--'
        mse_error_val='--'
    architecture_details=str(architecture['n_factor'])+architecture['arch']
    results=[ architecture_details, loss, batch_size, epochs, learning_rate,
            np.round(compress_ratio,4), n_bottleneck, mse_error_train, mse_error_val]
    
    if not FailedTraining and get_test_results:
        results_model=get_results_model(model,X_test)
        pyplot.legend()
        pyplot.savefig(savedir+f"{name_encoder}.png")
        pyplot.clf()
        # save full autoencoder model (without the decoder)
        model.save(savedir+f'{name_encoder}_AutoEncoder.h5')
        # define and save an encoder model (without the decoder)
        # if show_summary:
        #     encoder,decoder = get_encoder_decoder(model, "bottleneck")

        #     ## no need to save it loads corrupted after
        #     # encoder.save(f'{name_encoder}_encoder_compressratio_{np.round(n_bottleneck_ratio,4)}.h5')
        #     # define and save a decoder model (without the decoder)
        #     ## no need to save it loads corrupted after
        #     # decoder.save(f'{name_encoder}_decoder_compressratio_{np.round(n_bottleneck_ratio,4)}.h5')
        #     print("Full AutoEncoder")
        #     model.summary()
        #     print("Encoder")
        #     encoder.summary()
        #     print("Decoder")
        #     decoder.summary() 
    else:
        results_model=['--']*6        
    results+=results_model
    print(results,[type(i) for i in results])
    with open(results_filename, 'a') as f:
        text=""
        for i in range(len(results)):
            if isinstance(results[i], str):
                text+=f"{results[i]:>18} "
            else:
                text+=f"{round(results[i],8):18} "
                #print(results[i],isinstance(results[i], numbers.Number))
        f.write(text)
        f.write('\n')
    # transform this previous results variable in a dictionary with the variable names as keys:
    results_dict={ 'architecture_details':architecture_details, 'loss':loss, 'batch_size':batch_size,
                  'epochs':epochs, 'learning_rate':learning_rate, 'compress_ratio':np.round(compress_ratio,4),
                  'n_bottleneck':n_bottleneck, 'mse_error_train':mse_error_train, 'mse_error_val':mse_error_val}
    # results model dict has the get_results_model results
    results_model_dict = {'correlation':results_model[0],
                            'cosine':results_model[1],
                            'MAE':results_model[2],
                            'RMSE':results_model[3],
                            'r2':results_model[4],
                            'RMSE0':results_model[5],
                            }
    results_dict.update(results_model_dict)
    if return_results:
        return results_dict


def create_autoencoder(n_inputs : int = None,
                       layers_structure : list = None,
                       loss : str = 'mse',
                       lr : float = 0.001,
                       kl_loss_coeff : float = 5e-4,
                       type_architecture : str = 'default',
                       conv_reduction : int = 2, ## only if type_architecture is convoluted
                       asymmetric_loss_weights = None,
                       ):


    # Define the number of layers and number of neurons in each layer
    neurons_per_layer = layers_structure # [64, 32, 16, 8]
    num_layers = len(neurons_per_layer)
    # The input layer is the same as the output layer
    input_layer = Input(shape=(n_inputs,),name='input_enc')

    # Create the encoder layers
    # encoder_layers = []
    if type_architecture == 'default':
        for i in range(num_layers-1):
            if i == 0:
                e = Dense(neurons_per_layer[i])(input_layer)
                e = BatchNormalization()(e)
                e = ReLU()(e)
                # encoder_layers.append(Dense(neurons_per_layer[i], activation='relu')(BatchNormalization()(input_layer)))
            else:
                e = Dense(neurons_per_layer[i])(e)
                e = BatchNormalization()(e)
                e = ReLU()(e)
                # encoder_layers.append(Dense(neurons_per_layer[i], activation='relu')(BatchNormalization()(encoder_layers[i-1])))
                # encoder_layers.append(BatchNormalization()(Dense(neurons_per_layer[i], activation='relu')(encoder_layers[i-1])))

        # Create the decoder layers
        # decoder_layers = []
        for i in range(num_layers-1, -1, -1):   
            if i == num_layers-1:
                d = Dense(neurons_per_layer[i], name='bottleneck')(e)
                d = BatchNormalization()(d)
                d = ReLU()(d)
            else:
                d = Dense(neurons_per_layer[i])(d)
                d = BatchNormalization()(d)
                d = ReLU()(d)

    elif type_architecture == 'convoluted' :
        for i in range(num_layers-1):
            if i == 0:
                e = Dense(neurons_per_layer[i])(input_layer)
                e = Reshape((-1, 1))(e)
                e = Conv1D(filters=1,strides=2,kernel_size=int(neurons_per_layer[i]/(conv_reduction)))(e)
                e = BatchNormalization()(e)
                e = ReLU()(e)
                e = Flatten()(e)

            else:
                e = Dense(neurons_per_layer[i])(e)
                e = BatchNormalization()(e)
                e = ReLU()(e)

        # Create the decoder layers
        # decoder_layers = []
        for i in range(num_layers-1, -1, -1):   
            if i == num_layers-1:
                d = Dense(neurons_per_layer[i], name='bottleneck')(e)
                d = Reshape((-1, 1))(d)
                d = Conv1DTranspose(filters=1, kernel_size=int(neurons_per_layer[0]/conv_reduction),
                                    )(d)
                d = BatchNormalization()(d)
                d = ReLU()(d)
                d = Flatten()(d)
            else:
                d = Dense(neurons_per_layer[i])(d)
                d = BatchNormalization()(d)
                d = ReLU()(d)

    ## VAE architecture 
    ## https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
    elif type_architecture == 'VAE' :
        for i in range(num_layers-1):
            if i == 0:
                e = Dense(neurons_per_layer[i], name=f'dense_enc{i}')(input_layer)
                # e = Dropout(0.2)(e) # dropout layer to avoid overfitting
                e = BatchNormalization()(e)
                e = ReLU()(e)
                # encoder_layers.append(Dense(neurons_per_layer[i], activation='relu')(BatchNormalization()(input_layer)))
            else:
                e = Dense(neurons_per_layer[i], name=f'dense_enc{i}')(e)
                # e = Dropout(0.2)(e) # dropout layer to avoid overfitting
                e = BatchNormalization()(e)
                e = ReLU()(e)
                # encoder_layers.append(Dense(neurons_per_layer[i], activation='relu')(BatchNormalization()(encoder_layers[i-1])))
                # encoder_layers.append(BatchNormalization()(Dense(neurons_per_layer[i], activation='relu')(encoder_layers[i-1])))

        # Mean and variance of latent space
        latent_dim = neurons_per_layer[-1]
        z_mean = Dense(latent_dim, name='bottleneck_zmean')(e)
        z_log_var = Dense(latent_dim, name='bottleneck_zlog')(e)

        # Use sampling function to sample from latent space
        # encoder_output = Lambda(sampling, output_shape=(latent_dim,), 
        #                        name='bottleneck')([z_mean, z_log_var])
        encoder_output = SamplingLayer(name='bottleneck')([z_mean, z_log_var])
        encoder = Model(input_layer, encoder_output, name="encoder_model")

        decoder_input = Input(shape=(latent_dim,),name='input_dec')
        d = decoder_input 
        # Create the decoder layers
        # decoder_layers = []
        for i in range(num_layers-1, -1, -1):   
                d = Dense(neurons_per_layer[i], name=f'dense_dec{i}')(d)
                # d = Dropout(0.2)(d) # dropout layer to avoid overfitting
                d = BatchNormalization()(d)
                d = ReLU()(d)
                # dropout layer to avoid overfitting
                # e = Dropout(0.2)(e) 
    

        # The output layer is the same as the input layer
        # output_layer = Dense(n_inputs, activation='linear')(decoder_layers[-1])
        decoder_output = Dense(n_inputs, activation='linear', name='outputlayer')(d)
        decoder = Model(inputs=decoder_input, outputs=decoder_output, name="decoder_model")
        # Create the autoencoder model
        decoder_output = decoder(encoder_output) 
        if asymmetric_loss_weights is not None:
            autoencoder_output = AsymmetricVAELossLayer(loss=loss, asymmetric_loss_weights=asymmetric_loss_weights)([input_layer, decoder_output, z_mean, z_log_var])
        else:
            autoencoder_output = VAELossLayer(loss=loss, kl_loss_coeff=kl_loss_coeff)([input_layer, decoder_output, z_mean, z_log_var])
  
        autoencoder = Model(inputs=input_layer, outputs=autoencoder_output, name="VAE")
    
        # # Compile the model
        # if type_architecture == 'VAE' :
        # Add a custom loss function to account for the KL divergence
        # import tensorflow as tf
        # from tensorflow import keras
        # from tensorflow.keras import layers

        # def vae_loss(x, x_decoded_mean):
        #     ## Losses supported are binary_crossentropy, mean_squared_error, and log_cosh
        #     if loss == 'binary_crossentropy':
        #         xent_loss = tf.keras.losses.binary_crossentropy(x, x_decoded_mean)
        #     elif loss == 'mean_squared_error' or loss == 'mse':
        #         xent_loss = tf.keras.losses.mean_squared_error(x, x_decoded_mean)
        #     elif loss == 'log_cosh' or loss == 'logcosh':
        #         xent_loss = tf.math.log(tf.math.cosh(x - x_decoded_mean))
        #     else:
        #         raise ValueError('Unsupported loss function for VAE')

        #     kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        #     kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        #     kl_loss *= -0.5
        #     vae_loss = tf.keras.backend.mean(xent_loss + kl_loss_coeff*kl_loss)
        #     return vae_loss
        autoencoder.compile(optimizer='adam', loss=None, metrics=['mse'])
        autoencoder.summary()
        # return autoencoder, vae_loss    
        return autoencoder
  
    else:
        autoencoder.compile_model( optimizer=Adam(learning_rate=lr), loss = loss,
                                  metrics=['mse'])
        autoencoder.summary()
        return autoencoder

## for compatibility with previous version
TestEncoding = train_autoencoder
HyperParameterTestEncoding = hypertune_autoencoder

        
