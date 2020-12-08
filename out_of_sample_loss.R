source("data_preparation.R")
source("helper_functions.R")



recursive_prediction <- function(last_observed_years, subdata, gender, countries, country, timesteps, feature_dimension, model) {#, x_min, x_max){

        # Since the range of the years between the countries differ, we have to compute the last real observed year.
        last_year <- range(subdata$Year[which(data$Country == country)])[2]

        yearly_mse <- array(NA, c(last_year - last_observed_years))

        for (current_year in ((last_observed_years+1):last_year)){
                # Select only the necessary for the current year.
                data_current_year <- data_preprocessing(subdata[which(subdata$Year >= (current_year - timesteps)),], gender, country, timesteps, feature_dimension, current_year)

                # MinMaxScaler (with minimum and maximum from above)
                #x_test <- array(2*(data_current_year[[1]]-x_min)/(x_min-x_max)-1, dim(data_current_year[[1]]))
                x_test <- array(data_current_year[[1]], dim(data_current_year[[1]]))

                if (gender == "Female")
                        gender_index <- 0
                else
                        gender_index <- 1

                country_index <- get_country_index(country, countries)

                x_test <- list(x_test, rep(gender_index, dim(x_test)[1]), rep(country_index, dim(x_test)[1]))
                y_test <- - data_current_year[[2]]

                # TODO: Use exponential function like in the above github repository.
                # Predict the mortality rates for the current year.
                y_hat <- - as.vector(model %>% predict(x_test))

                # Calculate mean squared of the predictions of the current year.
                yearly_mse[current_year - last_observed_years] <- mean((y_hat - (-y_test))^2)

                # Replace the known log mortalities with the predicted ones.
                predicted <- subdata[which(subdata$Year == current_year),]
                keep <- subdata[which(subdata$Year != current_year),]
                predicted$log_mortality <- y_hat
                predicted$mortality <- exp(predicted$log_mortality)
                subdata <- rbind(keep, predicted)
                subdata <- subdata[order(Gender, Year, Age),]
        }

        list(subdata, yearly_mse)

}



out_of_sample_loss <- function(model, data, countries, timesteps, age_range, last_observed_year) {

        table <- matrix(0, length(countries) * 2, 2)
        colnames(table) <- c("MSE", "log MSE")
        row_names <- c()
        for(country in countries) {
                row_names <- c(row_names, paste(country, "Female", sep = " "))
                row_names <- c(row_names, paste(country, "Male", sep = " "))
        }
        rownames(table) <- row_names

        for (country in countries) {
                # Test data pre-processing.
                x_test_female <- data[which((data$Year > (last_observed_year - timesteps)) & (Gender == "Female") & (Country == country)),]
                y_test_female <- x_test_female[which(x_test_female$Year > last_observed_year),]
                x_test_male <- data[which((data$Year > (last_observed_year-timesteps)) & (Gender == "Male") & (Country == country)),]
                y_test_male <- x_test_male[which(x_test_male$Year > last_observed_year),]
                
                country_index <- get_country_index(country, countries)

                # Female
                prediction_and_mse <- recursive_prediction(last_observed_year, x_test_female, "Female", countries, country, timesteps, age_range, model) #, x_min, x_max)
                # Filter the predicted mortality rates.
                prediction <- prediction_and_mse[[1]][which(x_test_female$Year > last_observed_year),]
                table[2*country_index + 1, 1] <- mean((prediction$mortality - y_test_female$mortality)^2)
                table[2*country_index + 1, 2] <- mean((prediction$log_mortality - y_test_female$log_mortality)^2)

                # Male
                prediction_and_mse <- recursive_prediction(last_observed_year, x_test_male, "Male", countries, country, timesteps, age_range, model) #, x_min, x_max)
                # Filter the predicted mortality rates.
                prediction <- prediction_and_mse[[1]][which(x_test_male$Year > last_observed_year),]
                table[2*country_index + 2, 1] <- mean((prediction$mortality - y_test_male$mortality)^2)
                table[2*country_index + 2, 2] <- mean((prediction$log_mortality - y_test_male$log_mortality)^2)
        }

        return(table)

}