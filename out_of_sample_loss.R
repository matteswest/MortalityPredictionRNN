source("data_preparation.R")
source("helper_functions.R")



recursive_prediction <- function(last_observed_years, subdata, gender, countries, country, timesteps, feature_dimension, model, number_of_projected_years = 0) {#, x_min, x_max){

        # Since the range of the years between the countries differ, we have to compute the last real observed year.
        last_year <- range(subdata$Year[which(subdata$Country == country)])[2]

        yearly_mse <- array(NA, c(last_year - last_observed_years))

        for (current_year in ((last_observed_years+1):(last_year + number_of_projected_years))){

                # If mortality rates are projected the table does not include a row for the current year, so this row has to be added to the data table.
                if (current_year > last_year) {
                        new_row <- subdata[which(subdata$Year == (current_year - 1))]
                        new_row$Year <- current_year
                        subdata <- rbind(subdata, new_row)
                }

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

                # Replace the known log mortalities with the predicted ones.
                predicted <- subdata[which(subdata$Year == current_year),]
                keep <- subdata[which(subdata$Year != current_year),]
                predicted$log_mortality <- y_hat
                predicted$mortality <- exp(predicted$log_mortality)
                subdata <- rbind(keep, predicted)
                subdata <- subdata[order(Gender, Year, Age),]
        }

        list(subdata)

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
                recursive_pred <- recursive_prediction(last_observed_year, x_test_female, "Female", countries, country, timesteps, age_range, model) #, x_min, x_max)
                # Filter the predicted mortality rates.
                prediction <- recursive_pred[[1]][which(x_test_female$Year > last_observed_year),]
                table[2*country_index + 1, 1] <- mean((prediction$mortality - y_test_female$mortality)^2)
                table[2*country_index + 1, 2] <- mean((prediction$log_mortality - y_test_female$log_mortality)^2)

                # Male
                recursive_pred <- recursive_prediction(last_observed_year, x_test_male, "Male", countries, country, timesteps, age_range, model) #, x_min, x_max)
                # Filter the predicted mortality rates.
                prediction <- recursive_pred[[1]][which(x_test_male$Year > last_observed_year),]
                table[2*country_index + 2, 1] <- mean((prediction$mortality - y_test_male$mortality)^2)
                table[2*country_index + 2, 2] <- mean((prediction$log_mortality - y_test_male$log_mortality)^2)
        }

        return(table)

}



project_future_rates <- function(model, data, countries, timesteps, age_range, last_observed_year, number_of_projected_years) {

        country_projections <- vector("list", length(countries))
        list_index <- 1
        for (country in countries) {
                x_test_female <- data[which((data$Year > (last_observed_year - timesteps)) & (Gender == "Female") & (Country == country)),]
                y_test_female <- x_test_female[which(x_test_female$Year > last_observed_year),]
                x_test_male <- data[which((data$Year > (last_observed_year-timesteps)) & (Gender == "Male") & (Country == country)),]
                y_test_male <- x_test_male[which(x_test_male$Year > last_observed_year),]

                # Female
                recursive_pred_female <- recursive_prediction(last_observed_year, x_test_female, "Female", countries, country, timesteps, age_range, model, number_of_projected_years) #, x_min, x_max)
                # Male
                recursive_pred_male <- recursive_prediction(last_observed_year, x_test_male, "Male", countries, country, timesteps, age_range, model, number_of_projected_years) #, x_min, x_max)
                
                country_projections[[list_index]] <- list("Country" = country, "Female_Pred" = recursive_pred_female[[1]],
                                                          "Male_Pred" = recursive_pred_male[[1]])
                
                list_index <- list_index + 1
        }

        return(country_projections)

}