# Functions for data preprocessing
source("helper_functions.R")

# function that outputs training data set ( x_(t,x), Y_(t,x) )
data_preprocessing <- function(data, gender, country, timesteps, age_range, last_observed_years = 1999) { 

        mort_rates <- data[which((data$Gender == gender) & (data$Country == country)), c("Year", "Age", "log_mortality")] 
        mort_rates <- dcast(mort_rates, Year ~ Age, value.var = "log_mortality")
        if (last_observed_years > 2016){
                print("Str Mort_Rates:")
                print(last_observed_years)
                #print(str(mort_rates))
                #print(str(mort_rates[which(mort_rates$Year <= last_observed_years),]))
                print(mort_rates[which(mort_rates$Year == last_observed_years - 1)])
        }
        
        # selecting data
        train_rates <- as.matrix(mort_rates[which(mort_rates$Year <= last_observed_years),])
        # adding padding at the border
        (delta0 <- (age_range-1)/2)
        if (delta0>0){
                for (i in 1:delta0){
                        train_rates <- as.matrix(cbind(train_rates[,1], train_rates[,2], train_rates[,-1], train_rates[,ncol(train_rates)]))
                }
        }
        train_rates <- train_rates[,-1]
        (n_years <- nrow(train_rates)-(timesteps-1)-1)
        (n_ages <- ncol(train_rates)-(age_range-1)) 
        (n_train <- n_years * n_ages) # number of training samples
        xt_train <- array(NA, c(n_train, timesteps, age_range))
        yt_train <- array(NA, c(n_train))
        
        if (last_observed_years > 2015){
                print("Dimensions:")
                print(dim(train_rates))
                print(paste0("n_years:", n_years, " n_ages:", n_ages, " n_train:", n_train))
        }
        
        
        for (t0 in (1:n_years)){
                for (a0 in (1:n_ages)){
                        xt_train[(t0-1)*n_ages+a0,,] <- train_rates[t0:(t0 + timesteps - 1), a0:(a0+age_range-1)]
                        yt_train[(t0-1)*n_ages+a0] <- train_rates[t0 + timesteps, a0+delta0]
                }
        }
        list(xt_train, yt_train)

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