import os

import torch
import numpy as np
import math

from midi_hands.audio_model_builder import AudioModelBuilder
from midi_hands.preprocess_data_lstm_helpers import PreprocessDataLSTMHelpers
from midi_hands.utils import Utils
from torch.utils.tensorboard import SummaryWriter

class AudioModelRunner():
    """Incorporates training and testing the model"""
    @staticmethod
    def get_tensorboard_writer(comment=""):
        writer = SummaryWriter(comment=comment)

        return writer

    @staticmethod
    def get_batch_data_length(file_label):
        '''Get the number of files invovled in batch data'''
        file_path = f"preprocessed/{file_label}"
        directory_for_data = Utils.build_dir_path(file_path)
        all_files = os.listdir(directory_for_data)
        num_files = len(all_files)

        return num_files

    @staticmethod
    def get_batch_data(batch_index, file_label="train"):
        '''This is a callback used during training and ttesting, to get the data by batch'''
        return PreprocessDataLSTMHelpers.load_preprocessed_data(batch_index, file_label)
    
    @staticmethod
    def load_and_test_model(
        model_name="base",
        get_batch_data=None,
        get_batch_data_length=None,
        model_layers=None,
        lstm_model_params=None,
        optimization_params=None
    ):
        all_batches = AudioModelRunner.load_data(
            get_batch_data=get_batch_data,
            get_batch_data_length=get_batch_data_length
        )
        model, loss_function, _, _ = AudioModelBuilder.assemble_existing_lstm_model(
            model_name=model_name,
            model_layers=model_layers,
            lstm_model_params=lstm_model_params,
            optimization_params=optimization_params
        )
        model.eval()

        return AudioModelRunner.__get_model_avg_test_error(model, all_batches, loss_function)

    @staticmethod
    def resume_model_train(
        model_name="base",
        epochs_to_train=1,
        get_batch_data=None,
        get_batch_data_length=None,
        optimization_params=None,
        model_layers=None,
        lstm_model_params=None,
    ):
        '''Loads a model using pytorch's model state dictionary,
        trains the model for n epochs, and saves it to file'''
        model, loss_function, optimizer, epochs_so_far = AudioModelBuilder.assemble_existing_lstm_model(model_name)

        AudioModelRunner.__train_model_start(
            model,
            loss_function,
            optimizer,
            epochs_to_train=epochs_to_train,
            epochs_so_far=epochs_so_far,
            get_batch_data=get_batch_data,
            get_batch_data_length=get_batch_data_length,
            model_name=model_name,
            optimization_params=optimization_params,
            model_layers=model_layers,
            lstm_model_params=lstm_model_params
        )

    @staticmethod
    def create_and_train_model(
        model_name="base",
        epochs_to_train=0,
        get_batch_data=None,
        get_batch_data_length=None,
        lstm_model_params=None,
        optimization_params=None,
        model_layers=None
    ):
        model, loss_function, optimizer = AudioModelBuilder.create_lstm_model(
            model_name=model_name,
            lstm_model_params=lstm_model_params,
            optimization_params=optimization_params,
            model_layers=model_layers
        )
        AudioModelRunner.__train_model_start(
            model,
            loss_function,
            optimizer,
            epochs_to_train=epochs_to_train,
            epochs_so_far=0,
            get_batch_data=get_batch_data,
            get_batch_data_length=get_batch_data_length,
            model_name=model_name,
            optimization_params=optimization_params,
            model_layers=model_layers,
            lstm_model_params=lstm_model_params,
        )


        

    @staticmethod
    def __train_model_start(
        model=None,
        loss_function=None,
        optimizer=None,
        epochs_so_far=0,
        model_name="base",
        epochs_to_train=0,
        get_batch_data=None,
        get_batch_data_length=None,
        optimization_params=None,
        model_layers=None,
        lstm_model_params=None
    ):
        model_path = AudioModelBuilder.get_model_path(model_name)
        writer = AudioModelRunner.get_tensorboard_writer(comment=model_name)

        model_dict = AudioModelRunner.__train_model(
            model,
            loss_function,
            optimizer,
            epochs_to_train=epochs_to_train,
            epochs_so_far=epochs_so_far,
            get_batch_data=get_batch_data,
            get_batch_data_length=get_batch_data_length,
            model_name=model_name,
            optimization_params=optimization_params,
            writer=writer,
            model_layers=model_layers,
            lstm_model_params=lstm_model_params,
        )
        torch.save(model_dict, model_path)
        writer.add_graph(model, input_to_model=torch.rand((100, 16, 4)))

        writer.close()

    @staticmethod
    def load_data(
        get_batch_data=None,
        get_batch_data_length=None,
        file_label="train"
    ):
        batch_len = get_batch_data_length(file_label)
        all_batches = [get_batch_data(batch_num, file_label=file_label) for batch_num in range(0, batch_len)]

        return all_batches

    @staticmethod
    def __train_model(
        model,
        loss_function,
        optimizer,
        epochs_to_train=1,
        epochs_so_far=0,
        get_batch_data=None,
        get_batch_data_length=None,
        model_name="base",
        optimization_params=None,
        model_layers=None,
        lstm_model_params=None,
        writer=None
    ):
        '''Trains a model and return the state of the model (model_state_dict),
        the state of the loss and optimizer,
        and total cumulative epochs this model has run'''
        total_epochs = epochs_so_far + epochs_to_train

        # Used for keeping track over all the batches
        lowest_train_error = math.inf
        lowest_test_error = math.inf
        lowest_test_error_epochs = 0
        lowest_train_error_epochs = 0

        all_test_batches = AudioModelRunner.load_data(
            file_label="test",
            get_batch_data=get_batch_data,
            get_batch_data_length=get_batch_data_length
        )
        all_train_batches = AudioModelRunner.load_data(
            file_label="train",
            get_batch_data=get_batch_data,
            get_batch_data_length=get_batch_data_length
        )

        for t in range(epochs_to_train):
            current_epoch = t + 1
            training_error_by_batch = []
            accuracy_by_batch = []

            for batch in all_train_batches:
                (all_inputs, all_outputs) = batch

                batched_input_tensor = torch.tensor(all_inputs, dtype=torch.float, requires_grad=True)
                batched_output_tensor = torch.tensor(all_outputs, dtype=torch.float)

                # Forward pass
                y_pred = model(batched_input_tensor)

                # Calculate loss
                loss = loss_function(y_pred, batched_output_tensor)
                training_error_by_batch.append(loss.item())

                # reset optimizer state
                optimizer.zero_grad()

                # Backward pass
                loss.backward()
            
                # Update parameters
                optimizer.step()

                # Accuracy metric
                accuracy = model.get_accuracy(y_pred.clone(), torch.tensor(all_outputs, dtype=torch.float, requires_grad=False))
                accuracy_by_batch.append(accuracy)
            
            model.eval()

            avg_test_error, avg_train_error, avg_test_accuracy, avg_train_accuracy = AudioModelRunner.__get_metrics(
                model=model,
                all_test_batches=all_test_batches,
                loss_function=loss_function,
                accuracy_by_batch=accuracy_by_batch,
                training_error_by_batch=training_error_by_batch
            )

            writer.add_scalar('Loss/train', avg_train_error, current_epoch)
            writer.add_scalar('Loss/test', avg_test_error, current_epoch)
            writer.add_scalar('Accuracy/train', avg_train_accuracy, current_epoch)
            writer.add_scalar('Accuracy/test', avg_test_accuracy, current_epoch)

            # Track the lowest test/train to know which model to use later
            if (avg_test_error < lowest_test_error):
                lowest_test_error = avg_test_error
                lowest_test_error_epochs = current_epoch
            if (avg_train_error < lowest_train_error):
                lowest_train_error = avg_train_error
                lowest_train_error_epochs = current_epoch

            print(
                "Epoch ",
                current_epoch,
                " | Train error: ",
                avg_train_error,
                " | Test error: ",
                avg_test_error
            )

            print(
                "         ",
                "Train Accuracy: ",
                avg_train_accuracy,
                "Test Accuracy: ",
                avg_test_accuracy
            )

            AudioModelRunner.__record_model_state(
                model,
                loss_function=loss_function,
                optimizer=optimizer,
                epoch=epochs_so_far + current_epoch + 1,
                model_name=model_name
            )
            model.train()

        model.eval()

        # Track some global metrics
        writer.add_scalar('Lowest/train', lowest_train_error, lowest_train_error_epochs)
        writer.add_scalar('Lowest/test', lowest_test_error, lowest_test_error_epochs)
        hparams_dict = {
            **model.get_hyper_params(),
            **optimization_params,
        }
        metric_dict = {
            "hparam/train": lowest_train_error,
            "hparam/test": lowest_test_error,
            "hparam/test_epochs": lowest_test_error_epochs
        }

        writer.add_hparams(hparams_dict, metric_dict)
        
        AudioModelRunner.load_model_and_save_pickled(
            model_name=model_name,
            epochs=lowest_test_error_epochs,
            lstm_model_params=lstm_model_params,
            model_layers=model_layers,
            optimization_params=optimization_params
        )
        return {
            'epoch': total_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_function.state_dict(),
        }

    @staticmethod
    def __get_metrics(
        model=None,
        all_test_batches=None,
        loss_function=None,
        accuracy_by_batch=None,
        training_error_by_batch=None
    ):
        # Compute error for this epoch
        avg_test_error, avg_test_accuracy = AudioModelRunner.__get_model_avg_test_error(
            model,
            all_test_batches,
            loss_function
        )
        avg_train_accuracy = torch.mean(torch.tensor(accuracy_by_batch, dtype=torch.float, requires_grad=False))
        avg_train_error_tensor = torch.mean(torch.tensor(training_error_by_batch, dtype=torch.float, requires_grad=False))
        avg_train_error = avg_train_error_tensor.item()

        return (avg_test_error, avg_train_error, avg_test_accuracy, avg_train_accuracy)
    
    @staticmethod
    def __get_model_avg_test_error(
        model,
        all_batches,
        loss_function
    ):
        all_errs = []
        all_accuracies = []
        with torch.no_grad():
            for batch_input, batch_output in all_batches:
                batch_input_tensor = torch.tensor(batch_input, dtype=torch.float, requires_grad=False)
                batch_output_tensor = torch.tensor(batch_output, dtype=torch.float, requires_grad=False)
                result = model(batch_input_tensor)
                err = loss_function(result, batch_output_tensor)
                err_item = err.item()
                all_errs.append(err_item)
                accuracy = model.get_accuracy(result.clone(), batch_output_tensor.clone())
                all_accuracies.append(accuracy)

        average_error = torch.mean(torch.tensor(all_errs, dtype=torch.float, requires_grad=False))
        average_accuracy = torch.mean(torch.tensor(all_accuracies, dtype=torch.float, requires_grad=False))

        return average_error.item(), average_accuracy.item()

    @staticmethod
    def __record_model_state(
        model,
        loss_function=None,
        optimizer=None,
        epoch=0,
        model_name="base"
    ):
        model_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_function.state_dict(),
        }
        full_path_model_state = Utils.build_dir_path(
            f"train_output/{model_name}/progress/models/model_{model_name}_state_at_epoch_{str(epoch).zfill(4)}.pth"
        )
        torch.save(model_state, full_path_model_state)

    @staticmethod
    def load_model_and_save_pickled(
        model_name="",
        epochs=None,
        lstm_model_params=None,
        model_layers=None,
        optimization_params=None
    ):
        model, _, _, _ = AudioModelBuilder.assemble_existing_lstm_model(
            model_name=model_name,
            epochs=epochs,
            lstm_model_params=lstm_model_params,
            model_layers=model_layers,
            optimization_params=optimization_params
        )
        model.eval()
        AudioModelRunner.save_model_pickled_to_file(model=model, model_name=model_name)
        
    @staticmethod
    def save_model_pickled_to_file(model=None, model_name=""):
        model.eval()
        torch.save(model, Utils.build_dir_path(f"pickled/model_{model_name}.pt"))
