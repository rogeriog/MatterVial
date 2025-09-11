from typing import List,Union
import pandas as pd
import pickle 
from keras.models import load_model
from .autoencoder_setup import get_encoder_decoder, get_results_model, \
                               VAELossLayer, SamplingLayer

from keras.models import Model
import warnings
warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')
def filter_features_dataset(dataset = None,
        allowed_features : List[str] = None,
        allowed_features_file : str = None,
        mode : str = 'default',
                  ):
    if mode == "default":
        Xtofilter=dataset
    elif mode == "MODData":
        from modnet_gnn.preprocessing import MODData

        Xtofilter=dataset.df_featurized

    if allowed_features_file is not None:
        file_encoded_columns = open(allowed_features_file, 'r')
        lines = file_encoded_columns.readlines()
        columns_encoded=[line.strip('\n') for line in lines]
    elif allowed_features is not None:
        columns_encoded = allowed_features
    else:
        raise ValueError("Please provide a list of allowed features or a file with them.")
    ## Xtoencode needs to have all encoded columns in the scaler and autoencoder
    ## if not it will throw error. Please get the missing features.
    ## But columns that are in X but not in columns_encoded are discarded.
   
    Xtofilter=Xtofilter[[c for c in Xtofilter.columns if c in columns_encoded]]
    Xset=set(Xtofilter.columns)
    colset=set(columns_encoded)
    colmissing=list(colset-Xset)
    print(f"All feats missing: {colmissing}")
    tocompute=set([i.split('|')[0] for i in colmissing])
    print(f"You probably need to compute the following features: {tocompute}")
    if len(tocompute) != 0 :
        colset_feats=set([i.split('|')[0] for i in colset])
        if tocompute.issubset(colset_feats):
            ## in this case the features were calculated but there are specific
            ## properties missing, we will include those and fill with 0s for the encoder.
            for missing in colmissing:
                Xtofilter.loc[:, missing] = 0
        else:
            raise ValueError("Compute the aforementioned features before proceeding!")
    ## reorganizing columns in encoded columns
    Xtofilter = Xtofilter.reindex(columns_encoded, axis=1)

    if mode == "default":
        return Xtofilter
    elif mode == "MODData":
        dataset.df_featurized = Xtofilter
        return dataset


def encode_dataset(dataset  = None,
        scaler : str = None,
        columns_to_read : str = None,
        autoencoder : str = None,
        save_name : str = None,
        feat_prefix : str = "EncodedFeat",
        mode : str = "default",
        custom_objs : dict = None, 
        compile_model : bool = True,
        # for example: {'vae_loss':function}
        encoder_type : str = "regular",
                  ):
    if mode == "default":
        Xtoencode=dataset
        indexes=dataset.index
    elif mode == "MODData":
        Xtoencode=dataset.df_featurized
        indexes = dataset.df_featurized.index
    ## filtering features to encode dataset
    Xtoencode = filter_features_dataset(dataset=Xtoencode,allowed_features_file=columns_to_read)
    ## scaler data
    t=pickle.load(open(scaler,"rb"))
    Xtoencode = t.transform(Xtoencode)
    print('print Xtoencode:',Xtoencode)
    if custom_objs is not None:
        if compile_model == True:
            autoencoder = load_model(autoencoder, 
                                    custom_objects=custom_objs)
        else:
            autoencoder = load_model(autoencoder, 
                                    custom_objects=custom_objs,
                                    compile=False)
    else:
        if compile_model == True:
            autoencoder = load_model(autoencoder)
        else:
            autoencoder = load_model(autoencoder, compile=False)
    custom_objs = None
    if encoder_type == "regular":
        # if there is conflicting name this line may fix it.
        # The name "input_1" is used 2 times in the model. All layer names should be unique.
        # autoencoder.layers[0]._name='changed_input'
        encoder,decoder = get_encoder_decoder(autoencoder, "bottleneck")
        Xencoded=encoder.predict(Xtoencode)
        # autoencoder = load_model(autoencoder, 
        #                          custom_objects=custom_objs,
        #                          compile=False)
        Xencoded=pd.DataFrame(Xencoded, columns=[f"{feat_prefix}|{idx}" for idx in range(Xencoded.shape[1])],
                            index=indexes)
        if mode == "default":
            pickle.dump(Xencoded, open(save_name,'wb'))
        elif mode == "MODData":
            dataset.df_featurized=Xencoded
            dataset.save(save_name)
        print(Xencoded)
        print('Final shape:', Xencoded.shape)
        print('Summary of results:', get_results_model(autoencoder,Xtoencode))
        return Xencoded
    elif encoder_type == "variational":
        custom_objs={'SamplingLayer': SamplingLayer, 'VAELossLayer':VAELossLayer}
        autoencoder = load_model(autoencoder, 
                                custom_objects=custom_objs,
                                compile=False)
        encoder_layer_model = Model(inputs=autoencoder.input, outputs=autoencoder.layers[-3].output)
        Xencoded = encoder_layer_model.predict(Xtoencode, verbose=False)
        Xencoded=pd.DataFrame(Xencoded, columns=[f"{feat_prefix}|{idx}" for idx in range(Xencoded.shape[1])],
                    index=indexes)
        
        if mode == "default":
            pickle.dump(Xencoded, open(save_name,'wb'))
        elif mode == "MODData":
            dataset.df_featurized=Xencoded
            dataset.save(save_name)
        print(Xencoded)
        print('Final shape:', Xencoded.shape)
        print('Summary of results:', get_results_model(autoencoder,Xtoencode))
        return Xencoded

def decode_dataset(dataset = None,  
                   scaler: str = None,  
                   autoencoder: str = None,  
                   save_name: str = None,  
                   mode: str = "default",  
                   custom_objs: dict = None,  
                   compile_model: bool = True,  
                   encoder_type: str = "regular",  
                   columns_to_read: str = None,  
                   ):  
    """  
    Decodes a dataset from its latent representation using a trained autoencoder and scaler.  
  
    This function takes an encoded dataset, loads the corresponding autoencoder and scaler,  
    and applies the decoder and inverse scaling transformation to reconstruct the  
    original feature space.  
  
    Args:  
        dataset (Union[pd.DataFrame, 'MODData']): The encoded dataset to decode.  
            Can be a pandas DataFrame or a MODData object.  
        scaler (str): Path to the saved (pickled) scaler object used for the  
            original data.  
        autoencoder (str): Path to the saved Keras autoencoder model file (.h5).  
        save_name (str, optional): If provided, the path to save the decoded dataset.  
            For 'default' mode, it's saved as a pickle file. For 'MODData' mode,  
            the object's save method is called. Defaults to None.  
        mode (str, optional): Specifies the input dataset type. Use 'default' for a  
            pandas DataFrame and 'MODData' for a MODData object.  
            Defaults to "default".  
        custom_objs (dict, optional): Dictionary of custom objects required for loading  
            the Keras model (e.g., custom layers). Defaults to None.  
        compile_model (bool, optional): Whether to compile the model upon loading.  
            Set to False if the model has a custom loss function included in a layer.  
            Defaults to True.  
        encoder_type (str, optional): The type of autoencoder used. Can be 'regular'  
            or 'variational'. This determines how the decoder is extracted.  
            Defaults to "regular".  
        columns_to_read (str, optional): Path to a text file containing the names of the  
            original features, with one feature name per line. This is used to set  
            the column names of the output DataFrame. If not provided, the function  
            will attempt to get them from the scaler object.  
  
    Returns:  
        pd.DataFrame: A pandas DataFrame containing the decoded (reconstructed) data  
        in the original feature space.  
    """  
    # Handle input based on mode  
    if mode == "default":  
        Xto_decode = dataset  
        indexes = dataset.index  
    elif mode == "MODData":  
        # This import is conditional as it might not be available in all environments  
        from modnet_gnn.preprocessing import MODData  
        Xto_decode = dataset.df_featurized  
        indexes = dataset.df_featurized.index  
    else:  
        raise ValueError("Mode must be either 'default' or 'MODData'")  
  
    # Load the full autoencoder model  
    if custom_objs:  
        autoencoder_model = load_model(autoencoder,  
                                     custom_objects=custom_objs,  
                                     compile=compile_model)  
    elif encoder_type == "variational":  
        # VAEs require custom layer objects  
        custom_objs = {'SamplingLayer': SamplingLayer, 'VAELossLayer': VAELossLayer}  
        autoencoder_model = load_model(autoencoder,  
                                     custom_objects=custom_objs,  
                                     compile=False)  # VAEs with loss in a layer are not compiled with a loss  
    else:  
        autoencoder_model = load_model(autoencoder, compile=compile_model)  
  
    # Extract the decoder part of the model  
    if encoder_type == "regular":  
        # Use the provided utility function to split the model  
        try:  
            _, decoder = get_encoder_decoder(autoencoder_model, "bottleneck")  
        except Exception as e:  
            raise RuntimeError(f"Failed to extract decoder using get_encoder_decoder. Error: {e}")  
    elif encoder_type == "variational":  
        # For a VAE, manually reconstruct the decoder model.  
        # The decoder starts after the SamplingLayer and ends before the VAELossLayer.  
        try:  
            sampling_layer_index = [isinstance(layer, SamplingLayer) for layer in autoencoder_model.layers].index(True)  
        except ValueError:  
            raise ValueError("Could not find a 'SamplingLayer' in the provided variational autoencoder model.")  
  
        latent_dim_shape = autoencoder_model.layers[sampling_layer_index].output_shape[1:]  
        decoder_input = Input(shape=latent_dim_shape, name="decoder_input")  
  
        # Chain the subsequent layers to form the decoder  
        x = decoder_input  
        for layer in autoencoder_model.layers[sampling_layer_index + 1:]:  
            if not isinstance(layer, VAELossLayer):  
                x = layer(x)  
          
        decoder = Model(decoder_input, x, name="decoder")  
    else:  
        raise ValueError(f"Unsupported encoder_type: {encoder_type}. Choose 'regular' or 'variational'.")  
  
    # Use the decoder to reconstruct the scaled data  
    print("Decoding data...")  
    X_decoded_scaled = decoder.predict(Xto_decode)  
  
    # Load the scaler and inverse-transform the data  
    print("Inverse scaling data...")  
    with open(scaler, "rb") as f:  
        scaler_obj = pickle.load(f)  
      
    X_decoded_unscaled = scaler_obj.inverse_transform(X_decoded_scaled)  
  
    # Get original feature names to create the final DataFrame  
    original_columns = None  
    if columns_to_read:  
        with open(columns_to_read, 'r') as f:  
            original_columns = [line.strip() for line in f.readlines()]  
    elif hasattr(scaler_obj, 'get_feature_names_out'):  
        original_columns = scaler_obj.get_feature_names_out()  
    elif hasattr(scaler_obj, 'feature_names_in_'):  
        original_columns = scaler_obj.feature_names_in_  
      
    if original_columns is None:  
        raise ValueError("Could not determine original feature names. Please provide 'columns_to_read'.")  
  
    if X_decoded_unscaled.shape[1] != len(original_columns):  
        raise ValueError(  
            f"Shape mismatch: Decoded data has {X_decoded_unscaled.shape[1]} columns, "  
            f"but {len(original_columns)} column names were found/provided."  
        )  
  
    # Create the final DataFrame  
    df_decoded = pd.DataFrame(X_decoded_unscaled, columns=original_columns, index=indexes)  
  
    # Save the decoded data if a path is provided  
    if save_name:  
        print(f"Saving decoded data to {save_name}...")  
        if mode == "default":  
            with open(save_name, 'wb') as f:  
                pickle.dump(df_decoded, f)  
        elif mode == "MODData":  
            dataset.df_featurized = df_decoded  
            dataset.save(save_name)  
        print("Save complete.")  
  
    print('Final decoded shape:', df_decoded.shape)  
    return df_decoded