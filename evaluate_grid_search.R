# This script evaluates the models created by the grid search.

library(dplyr)
library(keras)
library(tfruns)
library(data.table)


source("out_of_sample_loss.R")
source("create_model.R")

use_kaggle <- TRUE

# fixed parameters
last_observed_year <- 2006
countries <- c("CHE", "DEUT", "DNK", "ESP", "FRATNP", "ITA", "JPN", "POL", "USA")

# Load data.
data <- fread("./data/mortality.csv")
# Convert gender and country to factor variables.
data$Gender <- as.factor(data$Gender)
data$Country <- as.factor(data$Country)
# Add column for mortality.
data$mortality <- exp(data$log_mortality)
# Filter relevant countries.
data <- data[which(data$Country %in% countries),]

# Load best model.
if (use_kaggle) {
        results <- ls_runs(order = metric_val_loss, decreasing= F, runs_dir = 'grid_search_kaggle')
        results <- select(results, -c(output))
        unit_sizes <- c(results[1,11], results[1,12], results[1,13], results[1,14], results[1,15])
        unit_sizes <- unit_sizes[1:results[1,10]]
        if (results[1,7] == "LSTM") {
                model <- create_lstm_model(c(results[1,8], results[1,9]), unit_sizes, results[1,17], results[1,18], results[1,19], 0)
        }
        else {
                model <- create_gru_model(c(results[1,8], results[1,9]), unit_sizes, results[1,17], results[1,18], results[1,19], 0)
        }
        #load_model_weights_hdf5(model, paste0("./", results[1,1], "/best_model_weights.h5"))
        load_model_weights_hdf5(model, paste0("./grid_search_kaggle/2020-12-09T15-47-35Z/best_model_weights.h5"))
} else {
        results <- ls_runs(order = metric_val_loss, decreasing= F, runs_dir = 'grid_search')
        results <- select(results, -c(output))
        path <- paste0("./", results[1,1], "/best_model.h5")
        model <- load_model_hdf5(path)
        summary(model)
}

# Write to xlsx.
writexl::write_xlsx(results, "./data/results.xlsx")

out_of_sample_loss(model, data, countries, results[1,8], results[1,9], last_observed_year)
# models to consider: 3, 5, 7