# This R Script serves as Input for the 'tuning_run()'-function which is called in rnn_model.R
# and specifies the flags for the construction of various models from which we want to choose 
# the best performing one.

source("create_model.R")

FLAGS <- flags(
        flag_string('model_type', 'LSTM'),
        flag_integer('timesteps', 10),
        flag_integer('age_range', 5),
        flag_integer('feature_dimension0', 20),
        flag_integer('feature_dimension1', 15),
        flag_integer('feature_dimension2', 10),
        #flag_numeric('dropout', 0.05),
        #flag_numeric('lr', 0.01),
        #flag_integer('patience', 35),
        #flag_integer('pats', 20),
        flag_integer('batchsize', 100), # maybe higher?
        flag_string('activation', 'tanh'),
        flag_string('recurrent_activation', 'tanh')
)

####### TODO: Data Preparation
# read RDS file

# construct unit sizes vector
unit_sizes <- c(FLAGS$feature_dimension0, FLAGS$feature_dimension1, FLAGS$feature_dimension2)

# Create the wanted model.
if (model_type == "LSTM") {
        model <- create_lstm_model(c(FLAGS$timesteps, FLAGS$age_range), unit_sizes, "tanh", "sigmoid", average_label)
} else
        model <- create_gru_model(c(FLAGS$timesteps, FLAGS$age_range), unit_sizes, "tanh", "sigmoid", average_label)


# compile model (or do compiling step in create_model.R)
######

# early stopping
early_stop <- callback_early_stopping(monitor = 'val_loss', patience = 25)

# adaptive learning rate
lr_reducer <- callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1,
                                            patience = 25, verbose = 0, mode = "min",
                                            min_delta = 1e-03, cooldown = 0, min_lr = 0)

# fit model
history <- model %>% fit(x = x_train, y = y_train, validation_split = 0.2, batch_size = 100, epochs = 100, verbose = 1, callbacks = callback_list)

# save model as hdf5-file
save_model_hdf5(model, 'model.h5')
