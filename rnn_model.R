# Import libraries
library(tidyverse)
library(data.table)
library(keras)

# import source codes
source("data_preparation.R")
source("create_lstm_model.R")

# Set parameters.
timesteps <- 10
feature_dimension0 <- 5
feature_dimension1 <- 20
feature_dimension2 <- 15
last_observed_year <- 2006
country <- "CHE"

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
data_female <- data_preprocessing(data, "Female", country, timesteps, feature_dimension0, last_observed_year)
data_male <- data_preprocessing(data, "Male", country, timesteps, feature_dimension0, last_observed_year)

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

model <- create_lstm_model(c(timesteps, feature_dimension0), c(feature_dimension0, feature_dimension1), "tanh", "tanh", average_label)
summary(model)

# Compile network.
model %>% compile(optimizer = "adam", loss = "mse", metrics = list("mae"))

# TODO: Use callbacks.
#CBs <- callback_model_checkpoint(file.name, monitor = "val_loss", verbose = 0,  save_best_only = TRUE, save_weights_only = TRUE)
{current_time <- Sys.time()
        history <- model %>% fit(x = x_train, y = y_train, validation_split = 0.2, batch_size = 100, epochs = 50, verbose = 1)
Sys.time() - current_time}
plot(history)

# Validation data pre-processing.
data2_female <- data[which((data$Year > (last_observed_year - timesteps)) & (Gender == "Female") & (Country == country)),]
x_test_female <- data2_female
y_test_female <- x_test_female[which(x_test_female$Year > last_observed_year),]
data2.male <- data[which((data$Year > (last_observed_year-timesteps)) & (Gender == "Male") & (Country == country)),]
x_test_male <- data2.male
y_test_male <- x_test_male[which(x_test_male$Year > last_observed_year),]

# calculating out-of-sample loss: LC is c(Female=0.6045, Male=1.8152)
# Female
prediction_result <- recursive_prediction(last_observed_year, data2_female, "Female", country, timesteps, feature_dimension0, model)
#vali <- prediction_result[[1]][which(data2_female$Year > last_observed_year),]
#mean((vali$mx-y_test_female$mx)^2)
# Male
prediction_result <- recursive_prediction(last_observed_year, data2.male, "Male", country, timesteps, feature_dimension0, model)
#vali <- prediction_result[[1]][which(data2.male$Year > last_observed_year),]
#mean((vali$mx-y_test_male$mx)^2)