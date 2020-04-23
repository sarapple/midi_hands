from audio_model_builder import AudioModelBuilder
from preprocess_midi_inference_data import PreprocessMIDIInferenceData
from utils import Utils

def load_model_for_inference(
    model_name="base",
    model_layers=None,
    lstm_model_params=None,
    optimization_params=None,
    epochs=1,
    midi_input_filename=None,
    midi_output_filename=None
):
    '''Load a model that was built previously, and run the test against it'''
    model, _, _, _ = AudioModelBuilder.assemble_existing_lstm_model(
        model_name=model_name,
        model_layers=model_layers,
        lstm_model_params=lstm_model_params,
        optimization_params=optimization_params,
        epochs=epochs
    )
    model.eval()
    midifile = PreprocessMIDIInferenceData.get_left_right_hands_midifile(
        model,
        midi_input_filename=midi_input_filename
    )
    Utils.build_dir_path(midi_output_filename)
    midifile.save(midi_output_filename)