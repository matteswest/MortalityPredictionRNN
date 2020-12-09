# Import libraries
library(tidyverse)
library(data.table)
library(keras)
library(tensorflow)
tensorflow::tf$random$set_seed(10)

# import source codes
source("data_preparation.R")
source("create_model.R")
source("out_of_sample_loss.R")

# Set variable parameters.
model_type <- "LSTM"
timesteps <- 10
age_range <- 5
feature_dimension0 <- 20
feature_dimension1 <- 15
feature_dimension2 <- 10

# fixed parameters
last_observed_year <- 2006
countries <- c("CHE", "DEUT", "DNK", "ESP", "FRATNP", "ITA", "JPN", "POL", "USA")

use_best_model <- TRUE

# Load data.
data <- fread("./data/mortality.csv")
# Convert gender and country to factor variables.
data$Gender <- as.factor(data$Gender)
data$Country <- as.factor(data$Country)
# Add column for mortality.
data$mortality <- exp(data$log_mortality)
# Filter relevant countries.
data <- data[which(data$Country %in% countries),]

# Shuffle the training data.
combined_training_set <- create_training_data(data, countries, timesteps, age_range, last_observed_year)
x_train <- combined_training_set[[1]]
y_train <- combined_training_set[[2]]
rm(combined_training_set)

# The mean of y_train will be used as starting value for the intercept weight as it leeds to 
# faster convergence.
average_label <- mean(y_train)

unit_sizes <- c(feature_dimension0, feature_dimension1, feature_dimension2)

# Create the wanted model.
if (model_type == "LSTM") {
        model <- create_lstm_model(c(timesteps, age_range), unit_sizes, "tanh", "sigmoid", average_label)
} else
        model <- create_gru_model(c(timesteps, age_range), unit_sizes, "tanh", "sigmoid", average_label)
summary(model)

# Compile network.
optimizer <- optimizer_adam()
model %>% compile(optimizer = optimizer, loss = "mse", metrics = list("mae"))

lr_reducer <- callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1,
                                            patience = 25, verbose = 0, mode = "min",
                                            min_delta = 1e-03, cooldown = 0, min_lr = 0)

callback_list <- list(lr_reducer)

# Name model
model_name <- paste0(model_type, length(unit_sizes),"_", age_range, "_", feature_dimension0, "_",
                    feature_dimension1, "_", feature_dimension2)
#file.name <- paste("./Model_Full_Param/best_model_", name.model, sep="")
file_name <- paste0("./CallBack/best_model_", model_name)

# define Callback to save best model w.r.t. loss value (mse)
CBs <- NULL
if (use_best_model) {
        CBs <- callback_model_checkpoint(file_name, monitor = "val_loss", verbose = 1,
                                         save_best_only = TRUE, save_weights_only = TRUE)
        callback_list <- c(callback_list, CBs)
}

# Fit model and measure time
{current_time <- Sys.time()
        history <- model %>% fit(x = x_train, y = y_train, validation_split = 0.2, batch_size = 100, epochs = 100, verbose = 1, callbacks = callback_list)
Sys.time() - current_time}
plot_loss(model_name, history[[2]]$val_loss, history[[2]]$loss)

if (use_best_model) load_model_weights_hdf5(model, file_name)

out_of_sample_loss(model, data, countries, timesteps, age_range, last_observed_year)

# Calculate in-sample loss
#mean((-as.vector(model %>% predict(x_train)) - (-y_train))^2)
