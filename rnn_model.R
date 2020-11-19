# Import libraries
library(tidyverse)
library(data.table)
library(keras)

# import source codes
source("data_preparation.R")
source("create_lstm_model.R")

# Set parameters.
timesteps <- 10
age_range <- 5
feature_dimension0 <- 20
feature_dimension1 <- 15
feature_dimension2 <- 10
last_observed_year <- 1999
country <- "CHE"

use_best_model <- TRUE

# Load data.
data <- fread("https://raw.githubusercontent.com/DeutscheAktuarvereinigung/Mortality_Modeling/master/mortality.csv")
# Convert gender and country to factor variables.
data$Gender <- as.factor(data$Gender)
data$Country <- as.factor(data$Country)
# Add column for mortality.
data$mortality <- exp(data$log_mortality)
# Filter relevant countries.
data <- data[which(data$Country %in% c("CHE", "DEUT", "DNK", "ESP", "FRATNP", "ITA", "JPN", "POL", "USA")),]

# Split data into female and male.
data_female <- data_preprocessing(data, "Female", country, timesteps, age_range, last_observed_year)
data_male <- data_preprocessing(data, "Male", country, timesteps, age_range, last_observed_year)

# Check if dimensions of male and female data match.
if ( (dim(data_female[[1]])[1] != dim(data_male[[1]])[1]) | (dim(data_female[[2]])[1] != dim(data_male[[2]])[1]) )
        stop("Shapes of female and male are not the same!")

# Merge female and male data into one set.
sample_size <- dim(data_female[[1]])[1]
x_train <- array(NA, dim=c(2*sample_size, dim(data_female[[1]])[c(2,3)]))
y_train <- array(NA, dim=c(2*sample_size))
gender_indicator <- rep(c(0,1), sample_size)
for (l in 1:sample_size){
        x_train[(l-1)*2+1,,] <- data_female[[1]][l,,]
        x_train[(l-1)*2+2,,] <- data_male[[1]][l,,]
        # Invert label sign.
        y_train[(l-1)*2+1] <- - data_female[[2]][l]
        y_train[(l-1)*2+2] <- - data_male[[2]][l]
}

# MinMaxScaler data pre-processing.
#x_min <- min(x_train)
#x_max <- max(x_train)
#x_train <- list(array(2*(x_train-x_min)/(x_min-x_max)-1, dim(x_train)), gender_indicator)
x_train <- list(x_train, gender_indicator)

# The mean of y_train will be used as starting value for the intercept weight as it leeds to 
# faster convergence.
average_label <- mean(y_train)

unit_sizes <- c(feature_dimension0, feature_dimension1, feature_dimension2)

model <- create_lstm_model(c(timesteps, age_range), unit_sizes, "tanh", "tanh", average_label)
summary(model)

# Compile network.
optimizer <- optimizer_adam()
model %>% compile(optimizer = optimizer, loss = "mse", metrics = list("mae"))

lr_reducer <- callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1,
                                            patience = 25, verbose = 0, mode = "min",
                                            min_delta = 1e-03, cooldown = 0, min_lr = 0)

callback_list <- list(lr_reducer)

# Name model
model_name <- paste0("LSTM", length(unit_sizes),"_", age_range, "_", feature_dimension0, "_",
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
        history <- model %>% fit(x = x_train, y = y_train, validation_split = 0.2, batch_size = 100, epochs = 50, verbose = 1, callbacks = callback_list)
Sys.time() - current_time}
plot(history)

# Validation data pre-processing.
data2_female <- data[which((data$Year > (last_observed_year - timesteps)) & (Gender == "Female") & (Country == country)),]
x_test_female <- data2_female
y_test_female <- x_test_female[which(x_test_female$Year > last_observed_year),]
data2_male <- data[which((data$Year > (last_observed_year-timesteps)) & (Gender == "Male") & (Country == country)),]
x_test_male <- data2_male
y_test_male <- x_test_male[which(x_test_male$Year > last_observed_year),]

# Calculate in-sample loss
if (use_best_model) load_model_weights_hdf5(model, file_name)
mean((exp(as.vector(model %>% predict(x_train))) - exp(y_train))^2)

# calculating out-of-sample loss: LC is c(Female=0.6045, Male=1.8152)
# Female
prediction_and_mse <- recursive_prediction(last_observed_year, data2_female, "Female", country, timesteps, age_range, model, x_min, x_max)
# Filter the predicted mortality rates.
prediction <- prediction_and_mse[[1]][which(data2_female$Year > last_observed_year),]
print("MSE female mortality: ")
mean((prediction$mortality - y_test_female$mortality)^2)
print("MSE female log_mortality: ")
mean((prediction$log_mortality - y_test_female$log_mortality)^2)

# Male
prediction_and_mse <- recursive_prediction(last_observed_year, data2_male, "Male", country, timesteps, age_range, model, x_min, x_max)
# Filter the predicted mortality rates.
prediction <- prediction_and_mse[[1]][which(data2_male$Year > last_observed_year),]
print("MSE male mortality: ")
mean((prediction$mortality - y_test_male$mortality)^2)
print("MSE male log_mortality: ")
mean((prediction$log_mortality - y_test_male$log_mortality)^2)