import torch
import torch.nn as nn


# class TwoLayerLSTM(nn.Module):
#     def __init__(self, input_size=(49, 1), hidden_layer_size1=128, hidden_layer_size2=1, output_size=(1, 1)):
#         super().__init__()
#         self.hidden_layer_size1 = hidden_layer_size1
#         self.hidden_layer_size2 = hidden_layer_size2
#         self.num_feature_out = output_size[1]
#
#         self.lstm1 = nn.LSTM(input_size[1], hidden_layer_size1)
#         self.lstm2 = nn.LSTM(hidden_layer_size1, hidden_layer_size2)
#
#         # Adjusted linear layer to match the output size
#         self.linear = nn.Linear(hidden_layer_size2, output_size[1])
#
#         self.hidden_cell1 = (torch.zeros(1, input_size[0], self.hidden_layer_size1),
#                              torch.zeros(1, input_size[0], self.hidden_layer_size1))
#
#         self.hidden_cell2 = (torch.zeros(1, input_size[0], self.hidden_layer_size2),
#                              torch.zeros(1, input_size[0], self.hidden_layer_size2))
#
#     def forward(self, input_seq):
#         print("LSTM model is called")
#         lstm_out1, self.hidden_cell1 = self.lstm1(input_seq, self.hidden_cell1)
#         lstm_out2, self.hidden_cell2 = self.lstm2(lstm_out1, self.hidden_cell2)
#
#
#         # Apply linear transformation to the entire sequence
#         linear_out = self.linear(lstm_out2)
#         #lstm_out2_last = lstm_out2[:, -1, :]  # Taking the output of the last time step
#         # Slicing the output of linear transformation
#         linear_out_last = linear_out[:, -1:, :]  # Taking the output of the last time step
#
#         #predictions = self.linear(lstm_out2_last)
#         #predictions = linear_out_last.view(input_seq.size(0), -1, self.num_feature_out)
#         return linear_out_last

class CustomLSTM(nn.Module):
    def __init__(self, input_features, hidden_size, num_layers, output_features):
        super(CustomLSTM, self).__init__()
        """
        input_size: The number of features in the input data.
        hidden_size: The number of features in the hidden state of the LSTM.
        num_layers: The number of LSTM layers.
        output_features: The number of output features (dimensionality of the output).
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_features = output_features

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_features, hidden_size, num_layers, batch_first=True)

        # Linear layer to map LSTM output to desired output size
        self.linear = nn.Linear(hidden_size, self.output_features)

    def forward(self, x, T2=1): # T2 how many time steps in the future [B,T2,feature_out]
        """
        x: The input tensor of shape [batch_size, time_steps_input, input_features].
        batch_size: The number of sequences in a batch.
        time_steps_input: The length of the input sequences.
        input_features: The number of features in the input.
        T2: The desired output sequence length.
        """
        batch_size, _, _ = x.size()

        # Initialize hidden state and cell state
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward pass through LSTM
        lstm_out, _ = self.lstm(x, (h_0, c_0))

        # Trim output sequence to desired length (T2)
        lstm_out_trimmed = lstm_out[:, :T2, :]

        # Pass through linear layer for final output (adjusting number of output features)
        output = self.linear(lstm_out_trimmed)

        return output

class ConvLSTM1D(nn.Module):
    def __init__(self, input_size, conv_filters, lstm_units, output_dim):
        super(ConvLSTM1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=conv_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=conv_filters, hidden_size=lstm_units, batch_first=True)
        self.fc = nn.Linear(lstm_units, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Adjust input shape for Conv1d
        x = self.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)  # Reshape for LSTM
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])  # Take the last time step's output
        return output





if __name__ == "__main__":
    # Define the sizes of the MLP layers
    # Example usage:
    # input_size = (49, 1)  # Define the input size as (sequence length, number of features)
    # output_size = (1, 1)  # Define the output size as (sequence length after prediction, number of output features)
    # model = TwoLayerLSTM(input_size=input_size, output_size=output_size)
    # Sample input data sizes

    # Example instantiation and usage
    input_feature = 10  # Number of input features
    hidden_size = 128  # Number of features in hidden state
    num_layers = 2  # Number of LSTM layers
    output_features = 1  # Desired output size

    # Create an instance of CustomLSTM
    lstm_model = CustomLSTM(input_feature, hidden_size, num_layers, output_features)
    n_params = sum(p.numel() for p in lstm_model.parameters())
    param_txt = str(n_params / 1e6)[:5] + "M"  # number of parameters in Milions
    print(param_txt)
    print("number of parameters: %.2fM" % (n_params / 1e6))

    # Sample input tensor
    batch_size = 450
    time_steps_input = 49
    input_data = torch.randn(batch_size, time_steps_input, input_feature)

    # Get model predictions
    output = lstm_model(input_data)
    print("Output Shape:", output.shape)