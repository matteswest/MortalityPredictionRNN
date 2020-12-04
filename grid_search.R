# Main

# Import libraries
library(data.table)
library(tfruns)

# Source R Scripts
source("data_preparation.R")

# fixed global parameters
last_observed_year <- 2006
countries <- c("CHE", "DEUT", "DNK", "ESP", "FRATNP", "ITA", "JPN", "POL", "USA")

# global parameters
set.seed(10)
parameters <- list(
        model_type = c("LSTM"),
        timesteps = c(3),#, 10, 15),
        age_range = c(7),
        layers = c(3),
        feature_dimension0 = c(20),
        feature_dimension1 = c(15),
        feature_dimension2 = c(10),
        feature_dimension3 = c(5),
        #feature_dimension4 = c(5, 10),
        batch_size = c(100),
        activation = c("tanh"),
        recurrent_activation = c("tanh"),
        output_activation = c("NULL")
)

# Create all combinations of training data
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
                        saveRDS(combined_training_set, file = paste0("./data/training_data/training_set_", timesteps, "_", age_range, ".rds"))
                } else {
                        print("Training data already exists.")
                }
                
        }
}


# delete grid_search directory if exists
if (file.exists("./grid_search")){
        unlink("./grid_search", recursive = TRUE)
}

runs <- tuning_run('hyperparameters.R', runs_dir = 'grid_search', sample = 1.0, flags = parameters)