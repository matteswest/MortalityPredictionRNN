# Functions for data preprocessing
source("helper_functions.R")

# function that outputs training data set ( x_(t,x), Y_(t,x) )
data_preprocessing <- function(data.raw, gender, country, timesteps, feature_dimension, last_observed_years = 1999) { 

        mort_rates <- data.raw[which((data.raw$Gender == gender) & (data.raw$Country == country)), c("Year", "Age", "log_mortality")] 
        mort_rates <- dcast(mort_rates, Year ~ Age, value.var = "log_mortality")
        # selecting data
        train_rates <- as.matrix(mort_rates[which(mort_rates$Year <= last_observed_years),])
        # adding padding at the border
        (delta0 <- (feature_dimension-1)/2)
        if (delta0>0){
                for (i in 1:delta0){
                        train_rates <- as.matrix(cbind(train_rates[,1], train_rates[,2], train_rates[,-1], train_rates[,ncol(train_rates)]))
                }
        }
        train_rates <- train_rates[,-1]
        (t1 <- nrow(train_rates)-(timesteps-1)-1)
        (a1 <- ncol(train_rates)-(feature_dimension-1)) 
        (n.train <- t1 * a1) # number of training samples
        xt.train <- array(NA, c(n.train, timesteps, feature_dimension))
        YT.train <- array(NA, c(n.train))
        for (t0 in (1:t1)){
                for (a0 in (1:a1)){
                        xt.train[(t0-1)*a1+a0,,] <- train_rates[t0:(t0 + timesteps - 1), a0:(a0+feature_dimension-1)]
                        YT.train[(t0-1)*a1+a0] <- train_rates[t0 + timesteps, a0+delta0]
                }
        }
        list(xt.train, YT.train)

}



create_training_data <- function(data, countries, timesteps, age_range, last_observed_year) {

        sample_sizes_country <- array(NA, dim = c(length(countries)))
        for (index in 1:length(countries)) {
                sample_sizes_country[index] <- (last_observed_year - range(data[which(data$Country == countries[index])]$Year)[1] - (timesteps - 1)) * length(unique(data$Age))
        }

        whole_sample_size <- sum(sample_sizes_country)

        # Calculate the new sample size for different number of used countries.
        x_train <- array(NA, dim = c(2 * whole_sample_size, timesteps, age_range))
        y_train <- array(NA, dim = c(2 * whole_sample_size))
        country_indicator <- array(NA, dim = c(2 * whole_sample_size))

        country_index <- 1
        stride <- 0
        for (country in countries) {
                # Split data into female and male.
                data_female <- data_preprocessing(data, "Female", country, timesteps, age_range, last_observed_year)
                data_male <- data_preprocessing(data, "Male", country, timesteps, age_range, last_observed_year)

                # Check if dimensions of male and female data match.
                if ( (dim(data_female[[1]])[1] != dim(data_male[[1]])[1]) | (dim(data_female[[2]])[1] != dim(data_male[[2]])[1]) )
                        stop("Shapes of female and male are not the same!")

                for (index in 1:sample_sizes_country[country_index]){
                        x_train[stride + (index-1) * 2 + 1,,] <- data_female[[1]][index,,]
                        x_train[stride + (index-1) * 2 + 2,,] <- data_male[[1]][index,,]
                        # Invert label sign.
                        y_train[stride + (index-1) * 2 + 1] <- - data_female[[2]][index]
                        y_train[stride + (index-1) * 2 + 2] <- - data_male[[2]][index]
                }
                country_indicator[(stride + 1) : (stride + 2 * sample_sizes_country[country_index])] <- country_index - 1

                stride <- stride + 2 * sample_sizes_country[country_index]
                country_index <- country_index + 1
        }
        gender_indicator <- rep(c(0,1), whole_sample_size)

        # MinMaxScaler data pre-processing.
        #x_min <- min(x_train)
        #x_max <- max(x_train)
        #x_train <- list(array(2*(x_train-x_min)/(x_min-x_max)-1, dim(x_train)), gender_indicator)
        x_train <- list(x_train, gender_indicator, country_indicator)

        # Shuffle the training data.
        #combined_training_set <- shuffle_data(x_train, y_train)
        combined_training_set <- list(x_train, y_train)
        combined_training_set

}