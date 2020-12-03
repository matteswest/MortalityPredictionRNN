# This script evaluates the models created by the grid search.

library(dplyr)
library(keras)
library(tfruns)
library(tensorflow)


# Load results and order them by validation loss.
results <- ls_runs(order = metric_val_loss, decreasing= F, runs_dir = 'grid_search')
results <- select(results, -c(output))

# Load best model.
model <- load_model_hdf5("best_model.h5")

# Write to xlsx.
#writexl::write_xlsx(results, "./data/results.xlsx")