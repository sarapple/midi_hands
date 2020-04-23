import torch

def lstm_tuple_data_extraction(x, layer):
    return layer(x)[0]

def dropout_relu_linear_data_extraction(x, layer):
    return layer(x)

def batch_norm_data_extraction(x, layer):
    return (
            layer(x.permute(0, 2, 1))
                .permute(0, 2, 1)
    )

class LSTMAudio(torch.nn.Module):
    def __init__(self, input_dim=4, hidden_dim=100, batch_size=30, output_dim=1, num_layers=1, dropout=0, model_layers=None):
        super(LSTMAudio, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout

        # self.model_layers_callbacks = []
        # self.lstm_1 = torch.nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        # self.lstm_2 = torch.nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)
        # self.lstm_3 = torch.nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)
        # self.relu = torch.nn.LeakyReLU()

        model_dict = {}
        self.model_layers = model_layers
        for layer in list(set(model_layers)):
            if (layer == "lstm_start"):
                model_dict[layer] = torch.nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
            elif (layer == "lstm"):
                model_dict[layer] = torch.nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)
            elif (layer == "dropout"):
                model_dict[layer] = torch.nn.Dropout(p=0.3)
            elif (layer == "relu"):
                model_dict[layer] = torch.nn.LeakyReLU()
            elif (layer == "batch_norm"):
                model_dict[layer] = torch.nn.BatchNorm1d(num_features=hidden_dim)
            elif(layer == "linear"):
                model_dict[layer] = torch.nn.Linear(hidden_dim, output_dim)
            else:
                raise Exception("Unhandled layer in LSTM model.")

        self.model = torch.nn.ModuleDict(model_dict)
        self.model_layers_callbacks = []
            # model_dict["lstm"]
        # Generate the inputs and output calls for each layerr
        for index, layer in enumerate(model_layers):
            if (layer == "lstm" or layer == "lstm_start"):
                if (layer == "lstm_start"):
                    lstm_input_dim = input_dim
                else:
                    lstm_input_dim = hidden_dim

                self.model_layers_callbacks.append((
                    lstm_tuple_data_extraction,
                    layer
                ))
            elif (layer == "dropout"):
                self.model_layers_callbacks.append((
                    dropout_relu_linear_data_extraction,
                    layer
                ))
            elif (layer == "relu"):
                self.model_layers_callbacks.append((
                    dropout_relu_linear_data_extraction,
                    layer
                ))
            elif (layer == "linear"):
                self.model_layers_callbacks.append((
                    dropout_relu_linear_data_extraction,
                    layer
                ))
            elif (layer == "batch_norm"):
                self.model_layers_callbacks.append((
                    batch_norm_data_extraction,
                    layer
                ))
        
            else:
                raise Exception("Unhandled layer in LSTM model.")

        
        # # Define the output layer
        # # self.linear = torch.nn.Linear(self.hidden_dim, output_dim)
        # self.model_layers = torch.nn.Sequential(**self.model_layers, torch.nn.Linear(self.hidden_dim, output_dim))

    def forward(self, input_data): # comes in batch_size x 16 x 6
        # import ipdb; ipdb.set_trace();
        x = input_data
        for layer_instance_call, layer in self.model_layers_callbacks:
            x = layer_instance_call(x, self.model[layer])

        # x, _ = self.model_layers(x)
        # x, _ = self.lstm_2(x)
        # x, _ = self.lstm_3(x)
        # x = self.relu(x)

        # y_pred = self.linear(x) # batch_size (batch size) x 16 (sequence) x 1 (output dim)
        
        return x
    
    def get_accuracy(self, outputs, truths):
        truths_flattened = truths.squeeze()
        outputs_flattened = outputs.squeeze()
        total_counter = 0
        correct_counter = 0
        # import ipdb; ipdb.set_trace();
        for batch_index, sequence in enumerate(outputs_flattened):
            for note_index, note_prediction in enumerate(sequence):
                total_counter += 1
                truth = truths_flattened[batch_index][note_index]
                if (
                    (truth >= 0 and note_prediction >= 0)
                    or (truth < 0 and note_prediction < 0)
                ):
                    correct_counter += 1
            
        return correct_counter / total_counter

            
    def get_hyper_params(self):
        return {
            "num_layers": self.num_layers,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "hidden_dim": self.hidden_dim,
            "input_dim": self.input_dim
        }
