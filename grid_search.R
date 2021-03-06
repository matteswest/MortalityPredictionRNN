# Main

# Import libraries
library(data.table)
library(tfruns)

# Source R Scripts
source("data_preparation.R")

# fixed global parameters
last_observed_year <- 2006
countries <- c("CHE", "DEUT", "DNK", "ESP", "FRATNP", "ITA", "JPN", "POL", "USA")

# Global parameters to be considered in grid search.
parameters <- list(
        model_type = c("LSTM"), # has to be either "LSTM" or "GRU", both not possible
        timesteps = c(5, 10, 15),
        age_range = c(5,7,9),
        layers = c(1,2,3,4),
        feature_dimension0 = c(20, 30, 40),
        feature_dimension1 = c(15, 20),
        feature_dimension2 = c(10, 15),
        feature_dimension3 = c(5, 10),
        batch_size = c(100, 200, 400),
        activation = c("tanh"),
        recurrent_activation = c("tanh", "sigmoid"),
        output_activation = c("exponential", "linear")
)

# Create all combinations of training data once so they do not have to be computed for each model.
# Load data.
data <- fread("./data/mortality.csv")
# Convert gender and country to factor variables.
data$Gender <- as.factor(data$Gender)
data$Country <- as.factor(data$Country)
# Add column for mortality.
data$mortality <- exp(data$log_mortality)
# Filter relevant countries.
data <- data[which(data$Country %in% countries),]

if (!file.exists("./data/training_data"))
        dir.create("./data/training_data")

for (timesteps in parameters$timesteps){
        for (age_range in parameters$age_range){
                
                if (!file.exists(paste0("./data/training_data/training_set_", timesteps, "_", age_range, ".rds"))) {
                        print("Create missing training data set...")
                        # Build training data
                        combined_training_set <- create_training_data(data, countries, timesteps, age_range, last_observed_year)
                        # Save as memory efficient RDS-file.
                        saveRDS(combined_training_set, file = paste0("./data/training_data/training_set_", timesteps, "_", age_range, ".rds"))
                } else {
                        print("Training data already exists.")
                }
                
        }
}

# Perform grid search. Set sample parameter to reduce number of combinations.
set.seed(10)
runs <- tuning_run('hyperparameters.R', runs_dir = paste0('grid_search_', parameters$model_type), sample = 0.01, flags = parameters)
