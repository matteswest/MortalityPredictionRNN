# Import libraries
library(tidyverse)
library(data.table)
library(keras)

# import source codes
source("data_preparation.R")
source("create_model.R")
source("shuffle_data.R")

# Set parameters.
model_type <- "GRU"
timesteps <- 10
age_range <- 5
feature_dimension0 <- 20
feature_dimension1 <- 15
feature_dimension2 <- 10
last_observed_year <- 1999
country <- "CHE"

use_best_model <- TRUE

# Load data.
data <- fread("./data/mortality.csv")
# Convert gender and country to factor variables.
data$Gender <- as.factor(data$Gender)
data$Country <- as.factor(data$Country)
# Add column for mortality.
data$mortality <- exp(data$log_mortality)
# Filter relevant countries.
data <- data[which(data$Country %in% c("CHE", "DEUT", "DNK", "ESP", "FRATNP", "ITA", "JPN", "POL", "USA")),]

# Shuffle the training data.
combined_training_set <- create_training_data(data, country, timesteps, age_range, last_observed_year)
x_train <- combined_training_set[[1]]
y_train <- combined_training_set[[2]]
rm(combined_training_set)

# The mean of y_train will be used as starting value for the intercept weight as it leeds to 
# faster convergence.
average_label <- mean(y_train)

unit_sizes <- c(feature_dimension0, feature_dimension1, feature_dimension2)

# Create the wanted model.
if (model_type == "LSTM") {
        model <- create_lstm_model(c(timesteps, age_range), unit_sizes, "tanh", "tanh", average_label)
} else
        model <- create_gru_model(c(timesteps, age_range), unit_sizes, "tanh", "tanh", average_label)
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
        CBs <- callback_model_checkpoint(file_name, monitor = "val_loss", verbose = 0,
                                         save_best_only = TRUE, save_weights_only = TRUE)
        callback_list <- c(callback_list, CBs)
}

# Fit model and measure time
{current_time <- Sys.time()
        history <- model %>% fit(x = x_train, y = y_train, validation_split = 0.2, batch_size = 100, epochs = 250, verbose = 1, callbacks = callback_list)
Sys.time() - current_time}
plot(history)

# Test data pre-processing.
x_test_female <- data[which((data$Year > (last_observed_year - timesteps)) & (Gender == "Female") & (Country == country)),]
y_test_female <- x_test_female[which(x_test_female$Year > last_observed_year),]
x_test_male <- data[which((data$Year > (last_observed_year-timesteps)) & (Gender == "Male") & (Country == country)),]
y_test_male <- x_test_male[which(x_test_male$Year > last_observed_year),]

# Calculate in-sample loss
if (use_best_model) load_model_weights_hdf5(model, file_name)
mean((exp(as.vector(model %>% predict(x_train))) - exp(y_train))^2)

# calculating out-of-sample loss: LC is c(Female=0.6045, Male=1.8152)
# Female
prediction_and_mse <- recursive_prediction(last_observed_year, x_test_female, "Female", country, timesteps, age_range, model) #, x_min, x_max)
# Filter the predicted mortality rates.
prediction <- prediction_and_mse[[1]][which(x_test_female$Year > last_observed_year),]
print("MSE female mortality: ")
mean((prediction$mortality - y_test_female$mortality)^2)
print("MSE female log_mortality: ")
mean((prediction$log_mortality - y_test_female$log_mortality)^2)

# Male
prediction_and_mse <- recursive_prediction(last_observed_year, x_test_male, "Male", country, timesteps, age_range, model) #, x_min, x_max)
# Filter the predicted mortality rates.
prediction <- prediction_and_mse[[1]][which(x_test_male$Year > last_observed_year),]
print("MSE male mortality: ")
mean((prediction$mortality - y_test_male$mortality)^2)
print("MSE male log_mortality: ")
mean((prediction$log_mortality - y_test_male$log_mortality)^2)