# Main

# Import libraries
library(data.table)

# fixed global parameters
last_observed_year <- 2006
countries <- c("CHE", "DEUT", "DNK", "ESP", "FRATNP", "ITA", "JPN", "POL", "USA")

# global parameters
set.seed(10)
parameters <- list(
        timesteps = c(5, 10, 15),
        age_range = c(5, 7, 9),
        layers = c(2, 3, 4),
        feature_dimension0 = c(20, 30, 40),
        feature_dimension1 = c(15, 25, 35),
        feature_dimension2 = c(10, 20, 30),
        feature_dimension3 = c(5, 15, 20),
        #feature_dimension4 = c(5, 10),
        batch_size = c(100),
        activation = c("tanh", "relu"),
        recurrent_activation = c("tanh", "sigmoid")
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


for (timesteps in parameters$timesteps){
        for (age_range in parameters$age_range){
                
                # Build training data
                combined_training_set <- create_training_data(data, countries, timesteps, age_range, last_observed_year)
                saveRDS(combined_training_set, file = paste0("./data/training_data/training_set_", timesteps, "_", age_range, ".rds"))
                
        }
}











