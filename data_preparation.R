# Functions for data preprocessing
source("helper_functions.R")

# This function creates the training data for one particular country-gender combination based on the number of timesteps,
# the number of surrounding ages and the last year in which mortality rates were observed.
data_preprocessing <- function(data, gender, country, timesteps, age_range, last_observed_years = 2006) { 

        # Get the log mortalities for the given gender and country.
        mort_rates <- data[which((data$Gender == gender) & (data$Country == country)), c("Year", "Age", "log_mortality")] 
        mort_rates <- dcast(mort_rates, Year ~ Age, value.var = "log_mortality")
        
        # Only use the years up to the last observed and year.
        train_rates <- as.matrix(mort_rates[which(mort_rates$Year <= last_observed_years),])

        # Since the model should also be able to make estimates for age 0 and 100, data must be added at the borders.
        (delta0 <- (age_range-1)/2)
        if (delta0>0){
                for (i in 1:delta0){
                        train_rates <- as.matrix(cbind(train_rates[,1], train_rates[,2], train_rates[,-1], train_rates[,ncol(train_rates)]))
                }
        }

        # Since the first column contains the years, remove it.
        train_rates <- train_rates[,-1]

        # Calculate the number of training samples for this one particular country-gender combination.
        (n_years <- nrow(train_rates)-(timesteps-1)-1)
        (n_ages <- ncol(train_rates)-(age_range-1)) 
        (n_train <- n_years * n_ages)

        # Create the training for this one particular country-gender combination.
        xt_train <- array(NA, c(n_train, timesteps, age_range))
        yt_train <- array(NA, c(n_train))
        for (t0 in (1:n_years)){
                for (a0 in (1:n_ages)){
                        xt_train[(t0-1)*n_ages+a0,,] <- train_rates[t0:(t0 + timesteps - 1), a0:(a0+age_range-1)]
                        yt_train[(t0-1)*n_ages+a0] <- train_rates[t0 + timesteps, a0+delta0]
                }
        }

        list(xt_train, yt_train)

}



# This function creates the dataset to be used for the training of the recurrent networks based on the following parameters:
# - data: the loaded data mortality set.
# - countries: List of countries to be used for training the model.
# - timesteps: Number of past years to be used to predict the next years mortality.
# - age_range: Number of surrounding ages to be added to the dataset (should be odd).
# - last_observed_year: Last year up to which the training dataset is created.
# To create a validation set the returned dataset has to be splitted.
create_training_data <- function(data, countries, timesteps, age_range, last_observed_year) {

        # Calculate the whole number of samples for one gender for every country.
        sample_sizes_country <- array(NA, dim = c(length(countries)))
        for (index in 1:length(countries)) {
                sample_sizes_country[index] <- (last_observed_year - range(data[which(data$Country == countries[index])]$Year)[1] - (timesteps - 1)) * length(unique(data$Age))
        }
        whole_sample_size <- sum(sample_sizes_country)

        # Initialize the arrays containing the training data set for every country and gender.
        x_train <- array(NA, dim = c(2 * whole_sample_size, timesteps, age_range))
        y_train <- array(NA, dim = c(2 * whole_sample_size))
        country_indicator <- array(NA, dim = c(2 * whole_sample_size))

        # Loop over every country and fill the initialized arrays.
        country_index <- 1
        stride <- 0
        for (country in countries) {
                # Split data into the two genders and create the preprocessed datasets based on the current country.
                data_female <- data_preprocessing(data, "Female", country, timesteps, age_range, last_observed_year)
                data_male <- data_preprocessing(data, "Male", country, timesteps, age_range, last_observed_year)

                # Check if dimensions of male and female data match.
                if ( (dim(data_female[[1]])[1] != dim(data_male[[1]])[1]) | (dim(data_female[[2]])[1] != dim(data_male[[2]])[1]) )
                        stop("Shapes of female and male are not the same!")

                # Fill the arrays for the global training dataset with the ones based on the current country.
                for (index in 1:sample_sizes_country[country_index]){
                        x_train[stride + (index-1) * 2 + 1,,] <- data_female[[1]][index,,]
                        x_train[stride + (index-1) * 2 + 2,,] <- data_male[[1]][index,,]
                        # Invert label sign, so that the exponential function can be used as the output activation in
                        # the neural network.
                        y_train[stride + (index-1) * 2 + 1] <- - data_female[[2]][index]
                        y_train[stride + (index-1) * 2 + 2] <- - data_male[[2]][index]
                }
                # Fill the array with the current country index.
                country_indicator[(stride + 1) : (stride + 2 * sample_sizes_country[country_index])] <- country_index - 1

                # Update the stride and country index.
                stride <- stride + 2 * sample_sizes_country[country_index]
                country_index <- country_index + 1
        }
        # Fill the array containing the gender index (0 = Female and 1 = Male).
        gender_indicator <- rep(c(0,1), whole_sample_size)

        # MinMaxScaler data pre-processing.
        #x_min <- min(x_train)
        #x_max <- max(x_train)
        #x_train <- list(array(2*(x_train-x_min)/(x_min-x_max)-1, dim(x_train)), gender_indicator)

        # Store the training data set in a list and return it.
        x_train <- list(x_train, gender_indicator, country_indicator)
        combined_training_set <- list(x_train, y_train)
        combined_training_set

}
