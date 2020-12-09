# This script evaluates the models created by the grid search.

library(dplyr)
library(keras)
library(tfruns)
library(data.table)

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

# Write to xlsx.
writexl::write_xlsx(results, "./data/results.xlsx")

# Load best model.
path <- paste0("./", results[1,1], "/best_model.h5")
model <- load_model_hdf5(path)
summary(model)

out_of_sample_loss(model, data, countries, results[1,8], results[1,9], last_observed_year)
# models to consider: 3, 5, 7





