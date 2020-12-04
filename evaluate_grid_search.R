# This script evaluates the models created by the grid search.

library(dplyr)
library(keras)
library(tfruns)

source("out_of_sample_loss.R")

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

# Load results and order them by validation loss.
results <- ls_runs(order = metric_val_loss, decreasing= F, runs_dir = 'grid_search')
results <- select(results, -c(output))

# Load best model.
model <- load_model_hdf5("best_model.h5")

out_of_sample_loss(model, data, countries, results$flag_timesteps, results$flag_age_range, last_observed_year)
# Write to xlsx.
#writexl::write_xlsx(results, "./data/results.xlsx")