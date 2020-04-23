import torch

from midi_hands.audio_model_builder import AudioModelBuilder
from midi_hands.audio_model_runner import AudioModelRunner

class AudioModelTester():
    '''Similar to the AudioModelRunner, but training is severely simplified for sanity checking'''
    @staticmethod
    def sanity_check_test_model(
        model_name="sanity",
        get_batch_data=None,
        file_label="test",
        model_layers=None,
        lstm_model_params=None,
        optimization_params=None
    ):
        '''Sanity check the model against sanity built net'''
        (batch_zero_input, batch_zero_output) = get_batch_data(0, file_label)
        (batch_one_input, batch_one_output) = get_batch_data(1, file_label)
        model, _, _, _epochs_so_far = AudioModelBuilder.assemble_existing_lstm_model(
            model_name=model_name,
            model_layers=model_layers,
            lstm_model_params=lstm_model_params,
            optimization_params=optimization_params
        )
        model.eval()

        result_zero = model(torch.tensor([batch_zero_input[0]], dtype=torch.float, requires_grad=False))
        result_one = model(torch.tensor([batch_one_input[0]], dtype=torch.float))

        print(result_zero[0])
        print(result_one[0])
        print(batch_zero_output[0])
        print(batch_one_output[0])

    @staticmethod
    def sanity_check_build_model(
        model_name="sanity",
        epochs_to_train=0,
        get_batch_data=None,
        get_batch_data_length=None,
        lstm_model_params=None,
        optimization_params=None,
        model_layers=None
    ):
        '''Sanity check and build a very simple net'''
        file_label = "train"
        batch_zero = get_batch_data(0, file_label)
        batch_one = get_batch_data(1, file_label)

        def get_batch_data_sanity(batch_index, file_label=file_label):
            if batch_index == 0:
                return batch_zero
            else:
                return batch_one

        def get_batch_data_length_sanity(file_label):
            return 2
    
        AudioModelRunner.create_and_train_model(
            model_name=model_name,
            epochs_to_train=epochs_to_train,
            get_batch_data=get_batch_data_sanity,
            get_batch_data_length=get_batch_data_length_sanity,
            lstm_model_params=lstm_model_params,
            optimization_params=optimization_params,
            model_layers=model_layers
        )
