library(keras)



# The following function creates a LSTM model based on the given parameters:
# - input_shape: (timesteps, feature_dimension).
# - unit_sizes: number of units in a LSTM-layer, the length of unit_sizes corresponds to the 
#               number of LSTM-layers.
# - average_label: is used as an initialization for the weight in the last dense layer for faster convergence.
create_lstm_model <- function(input_shape, unit_sizes, activation, recurrent_activation, average_label) {

        input <- layer_input(shape = input_shape, dtype = "float32")
        input_gender <- layer_input(shape = c(1), dtype = "float32")
        input_country <- layer_input(shape = c(1), dtype = "float32")

        current_output <- input

        if (length(unit_sizes) > 1) {
                for (layer_number in 1:(length(unit_sizes) - 1)) {
                        current_output <- layer_lstm(units = unit_sizes[layer_number], 
                                                     activation = activation, 
                                                     recurrent_activation = recurrent_activation, 
                                                     return_sequences = TRUE)(current_output)
                        #current_output <- layer_batch_normalization()(current_output)
                }
        }

        # For the last LSTM-layer the output is not return as sequences.
        current_output <- layer_lstm(units = unit_sizes[length(unit_sizes)], activation = activation, recurrent_activation = recurrent_activation, return_sequences = FALSE)(current_output)
        #current_output <- layer_batch_normalization()(current_output)

        # Concatenate LSTM result and gender indicator.
        current_output <- layer_concatenate(list(current_output, input_gender))
        current_output <- layer_concatenate(list(current_output, input_country))
        output <- layer_dense(units = 1, activation = k_exp, weights = list(array(0, dim = c(unit_sizes[length(unit_sizes)] + 2, 1)), array(log(average_label) ,dim = c(1))))(current_output)

        model <- keras_model(inputs = list(input, input_gender, input_country), outputs = c(output))

}



# The following function creates a GRU model based on the given parameters:
# - input_shape: (timesteps, feature_dimension).
# - unit_sizes: number of units in a GRU-layer, the length of unit_sizes corresponds to the 
#               number of GRU-layers.
# - average_label: is used as an initialization for the weight in the last dense layer for faster convergence.
create_gru_model <- function(input_shape, unit_sizes, activation, recurrent_activation, average_label) {

        input <- layer_input(shape = input_shape, dtype = "float32")
        input_gender <- layer_input(shape = c(1), dtype = "float32")
        input_country <- layer_input(shape = c(1), dtype = "float32")

        current_output <- input

        if (length(unit_sizes) > 1) {
                for (layer_number in 1:(length(unit_sizes) - 1)) {
                        current_output <- layer_gru(units = unit_sizes[layer_number], 
                                                     activation = activation, 
                                                     recurrent_activation = recurrent_activation, 
                                                     return_sequences = TRUE)(current_output)
                        #current_output <- layer_batch_normalization()(current_output)
                }
        }

        # For the last LSTM-layer the output is not return as sequences.
        current_output <- layer_gru(units = unit_sizes[length(unit_sizes)], activation = activation, recurrent_activation = recurrent_activation, return_sequences = FALSE)(current_output)
        #current_output <- layer_batch_normalization()(current_output)

        # Concatenate LSTM result and gender indicator.
        current_output <- layer_concatenate(list(current_output, input_gender))
        current_output <- layer_concatenate(list(current_output, input_country))
        output <- layer_dense(units = 1, activation = k_exp, weights = list(array(0, dim = c(unit_sizes[length(unit_sizes)] + 2, 1)), array(log(average_label) ,dim = c(1))))(current_output)

        model <- keras_model(inputs = list(input, input_gender, input_country), outputs = c(output))

}