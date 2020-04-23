import torch

from utils import Utils
from lstm_audio import LSTMAudio
# Do not import AudioModelRunner. The runner should always import the builder.

class AudioModelBuilder():
    """Assembles parts needed to build a model. Model, hyperparameters, optimizers, etc"""

    @staticmethod
    def get_model_path(model_name="base"):
        """Build the file path and intermediate directories needed for given model path"""
        return Utils.build_dir_path(f"train_output/{model_name}/completed/models/model_{model_name}.pth")
    
    @staticmethod
    def get_model_path_from_progress(model_name="base", epochs=0):
        """Build the file path and intermediate directories needed for given model path for models in progress"""

        return 'output/' + Utils.build_dir_path(f"train_output/{model_name}/progress/models/model_{model_name}_state_at_epoch_{str(epochs).zfill(4)}.pth")
    
    @staticmethod
    def get_hyper_parameters():
        '''Get the shared lstm instance variables (find a way to move this to the state_dict)'''
        return {
            "lstm_model_params": {
                "input_dim": 4, # number of features
                "hidden_dim": 200, # of features in the hidden state
                "batch_size": 30,
                # size of hidden layer can be larger/smaller than your sequence (number of notes)
                "num_layers": 2,
                "output_dim": 1, # input dim of 6 gets squished down to 1 dimension of handedness
                "dropout": 0.1
            },
            "optimization_params": {
                "learning_rate": 0.001
            }
        }

    @staticmethod
    def get_model_layers():
        '''Get the shared lstm instance variables (find a way to move this to the state_dict)'''
        return [
            "lstm",
            "dropout",
            "relu",
            "batch_norm"
        ]
    
    @staticmethod
    def create_lstm_model(
        model_name="base",
        lstm_model_params=None,
        optimization_params=None,
        model_layers=None
    ):
        '''Creates a new model from assembled parts'''

        model, loss_function, optimizer = AudioModelBuilder.__get_model_parts(
            lstm_model_params=lstm_model_params,
            optimization_params=optimization_params,
            model_layers=model_layers
        )
        
        model_dict = {
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_function.state_dict(),
        }

        model_path = AudioModelBuilder.get_model_path(model_name)
        torch.save(model_dict, model_path)

        return (
            model,
            loss_function,
            optimizer
        )
    
    @staticmethod
    def assemble_existing_lstm_model(
        model_name="base",
        epochs=None,
        model_layers=None,
        lstm_model_params=None,
        optimization_params=None,
        model_path=None
    ):
        '''Loads a model using pytorch's model state dictionary,
        and returns thte model, loss, optimizer, and epochs so far'''
        if (model_path is None):
            model_path = AudioModelBuilder.get_model_path(model_name) if epochs is None else AudioModelBuilder.get_model_path_from_progress(model_name, epochs=epochs)
        
        model, loss_function, optimizer = AudioModelBuilder.__get_model_parts(
            model_layers=model_layers,
            lstm_model_params=lstm_model_params,
            optimization_params=optimization_params
        )
        checkpoint = torch.load(model_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        loss_function.load_state_dict(checkpoint['loss'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs_so_far = checkpoint['epoch']

        return model, loss_function, optimizer, epochs_so_far

    @staticmethod
    def __get_loss_function():
        '''
        The loss function used by LSTM_Audio
        Currently this loss function is used outside of the state of the class.
        TODO: See if we can integrate this better to the state of the LSTM_Audio class'''
        # weight the left hand more since we have fewer of those
        # loss = torch.nn.MSELoss()
        return torch.nn.MSELoss()

    @staticmethod
    def __get_optimizer(model, optimization_params=None):
        '''
        The optimizer used by LSTM_Audio
        Currently this optimizer is used outside of the state of the class.
        TODO: See if we can integrate this better to the state of the LSTM_Audio class'''
        return torch.optim.Adam(model.parameters(), lr=optimization_params["learning_rate"])

    @staticmethod
    def __get_model_parts(lstm_model_params=None, optimization_params=None, model_layers=None):
        '''Get the model, loss function and optimizer'''
        model = LSTMAudio(
            **lstm_model_params,
            model_layers=model_layers
        )
        loss_function = AudioModelBuilder.__get_loss_function()
        optimizer = AudioModelBuilder.__get_optimizer(model, optimization_params=optimization_params)

        return (
            model,
            loss_function,
            optimizer
        )
