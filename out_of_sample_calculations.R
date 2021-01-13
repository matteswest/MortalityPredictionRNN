source("data_preparation.R")
source("helper_functions.R")



# Recursively apply the model to predict the mortality rates for the future and unknown years.
recursive_prediction <- function(last_observed_years, subdata, gender, countries, country, timesteps, feature_dimension, model, number_of_projected_years = 0) {#, x_min, x_max){

        # Since the range of the years between the countries differ, we have to compute the last real observed year.
        last_year <- range(subdata$Year[which(subdata$Country == country)])[2]

        for (current_year in ((last_observed_years+1):(last_year + number_of_projected_years))){
                # If mortality rates are projected the table does not include a row for the current year, so this row has to be added to the data table.
                if (current_year > last_year) {
                        new_row <- subdata[which(subdata$Year == (current_year - 1))]
                        new_row$Year <- current_year
                        subdata <- rbind(subdata, new_row)
                }

                # Select only the necessary data for the current year (current_year - timesteps until current_year).
                data_current_year <- data_preprocessing(subdata[which(subdata$Year >= (current_year - timesteps)),], gender, country, timesteps, feature_dimension, current_year)

                # MinMaxScaler (with minimum and maximum from above)
                #x_test <- array(2*(data_current_year[[1]]-x_min)/(x_min-x_max)-1, dim(data_current_year[[1]]))
                
                # Get mortality rates for the prediction of the current year.
                x_test <- array(data_current_year[[1]], dim(data_current_year[[1]]))

                if (gender == "Female")
                        gender_index <- 0
                else
                        gender_index <- 1

                country_index <- get_country_index(country, countries)

                # Create the complete test set for the prediction of the current year.
                x_test <- list(x_test, rep(gender_index, dim(x_test)[1]), rep(country_index, dim(x_test)[1]))

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



# Calculate the out of sample loss for a given list of countries using one or multiple models.
out_of_sample_loss <- function(models, data, countries, timesteps, age_range, last_observed_year) {

        # Create table for the mean squared errors to be returned.
        table <- matrix(0, length(countries) * 2, 3)
        colnames(table) <- c("MSE rec", "MSE ff", "MSE lc")
        row_names <- c()
        for(country in countries) {
                row_names <- c(row_names, paste(country, "Female", sep = " "))
                row_names <- c(row_names, paste(country, "Male", sep = " "))
        }
        rownames(table) <- row_names

        # MSEs of the deep feed forward neural network from 
        # https://github.com/DeutscheAktuarvereinigung/Mortality_Modeling/blob/master/DAV%20Use%20Case%20Mortality%20Modeling-Final_V3.ipynb.
        table[1, 2] <- 1.465979e-04
        table[2, 2] <- 1.858336e-04
        table[3, 2] <- 1.290387e-04
        table[4, 2] <- 1.065868e-04
        table[5, 2] <- 6.290338e-05
        table[6, 2] <- 3.539380e-04
        table[7, 2] <- 2.241446e-05
        table[8, 2] <- 4.092471e-05
        table[9, 2] <- 2.927603e-05
        table[10, 2] <- 5.463496e-05
        table[11, 2] <- 2.432748e-05	
        table[12, 2] <- 3.619865e-05
        table[13, 2] <- 1.075894e-04
        table[14, 2] <- 1.612694e-04
        table[15, 2] <- 3.363432e-05
        table[16, 2] <- 1.002263e-04
        table[17, 2] <- 8.846130e-06	
        table[18, 2] <- 2.190878e-05

        # MSEs of the Lee-Carter Model from 
        # https://github.com/DeutscheAktuarvereinigung/Mortality_Modeling/blob/master/DAV%20Use%20Case%20Mortality%20Modeling-Final_V3.ipynb.
        table[1, 3] <- 8.064493e-05
        table[2, 3] <- 1.823052e-04
        table[3, 3] <- 8.318317e-05
        table[4, 3] <- 1.115609e-04
        table[5, 3] <- 8.566356e-05
        table[6, 3] <- 3.834106e-04
        table[7, 3] <- 6.919801e-05
        table[8, 3] <- 2.033477e-04
        table[9, 3] <- 4.276970e-05
        table[10, 3] <- 7.958913e-05
        table[11, 3] <- 1.169243e-05	
        table[12, 3] <- 4.942109e-05
        table[13, 3] <- 4.705808e-05
        table[14, 3] <- 2.499710e-05
        table[15, 3] <- 3.239788e-04
        table[16, 3] <- 3.563289e-04
        table[17, 3] <- 1.042052e-05	
        table[18, 3] <- 7.230708e-05

        # Apply the recursive prediction for all country-gender combinations.
        for (country in countries) {
                predicted_mortality_female <- replicate(101 * (range(data[which(data$Country == country)]$Year)[2] - last_observed_year), 0.0)
                predicted_mortality_male <- replicate(101 * (range(data[which(data$Country == country)]$Year)[2] - last_observed_year), 0.0)
                
                # Use an ensemble of models to predict the mortality rates for the current country.
                for (model_index in 1:length(models)) {
                        # Test data pre-processing.
                        x_test_female <- data[which((data$Year > (last_observed_year - timesteps[[model_index]])) & (Gender == "Female") & (Country == country)),]
                        y_test_female <- x_test_female[which(x_test_female$Year > last_observed_year),]
                        x_test_male <- data[which((data$Year > (last_observed_year - timesteps[[model_index]])) & (Gender == "Male") & (Country == country)),]
                        y_test_male <- x_test_male[which(x_test_male$Year > last_observed_year),]

                        # Female
                        recursive_pred <- recursive_prediction(last_observed_year, x_test_female, "Female", countries, country, timesteps[[model_index]], age_range[[model_index]], models[[model_index]]) #, x_min, x_max)
                        # Filter the predicted mortality rates.
                        prediction <- recursive_pred[[1]][which(x_test_female$Year > last_observed_year),]
                        # Sum up the predictions.
                        predicted_mortality_female <- prediction$mortality + predicted_mortality_female

                        # Male
                        recursive_pred <- recursive_prediction(last_observed_year, x_test_male, "Male", countries, country, timesteps[[model_index]], age_range[[model_index]], models[[model_index]]) #, x_min, x_max)
                        # Filter the predicted mortality rates.
                        prediction <- recursive_pred[[1]][which(x_test_male$Year > last_observed_year),]
                        # Sum up the predictions.
                        predicted_mortality_male <- prediction$mortality + predicted_mortality_male
                }
                country_index <- get_country_index(country, countries)

                # Compute arithmetic mean.
                predicted_mortality_female <- predicted_mortality_female / length(models)
                predicted_mortality_male <- predicted_mortality_male / length(models)

                # Write mean squared errors into the table.
                table[2*country_index + 1, 1] <- mean((predicted_mortality_female - y_test_female$mortality)^2)
                table[2*country_index + 2, 1] <- mean((predicted_mortality_male - y_test_male$mortality)^2)
        }

        print(paste0("Average MSE for recurrent net: ", mean(table[1:(2*length(countries)), 1])))
        print(paste0("Average MSE for feedforward net: ", mean(table[1:(2*length(countries)), 2])))
        print(paste0("Average MSE for Lee-Carter model: ", mean(table[1:(2*length(countries)), 3])))

        print(table)

        print(paste0("The predictions of the recurrent network are better than those of the feedforward net in ", sum(table[1:(2*length(countries)), 1] < table[1:(2*length(countries)), 2]), 
        " out of ", 2*length(countries), " country-gender groups of mortality rates."))
        print(paste0("The predictions of the recurrent network are better than those of the Lee-Carter model in ", sum(table[1:(2*length(countries)), 1] < table[1:(2*length(countries)), 3]), 
        " out of ", 2*length(countries), " country-gender groups of mortality rates."))

        return(table)

}



# Project the mortality rates for every given country for the next years.
project_future_rates <- function(model, data, countries, timesteps, age_range, last_observed_year, number_of_projected_years) {

        # Loop over every given country to calculate the prediction for the next number_of_projected_years.
        country_projections <- vector("list", length(countries))
        list_index <- 1
        for (country in countries) {
                # Get relevant data.
                x_test_female <- data[which((data$Year > (last_observed_year - timesteps)) & (Gender == "Female") & (Country == country)),]
                y_test_female <- x_test_female[which(x_test_female$Year > last_observed_year),]
                x_test_male <- data[which((data$Year > (last_observed_year-timesteps)) & (Gender == "Male") & (Country == country)),]
                y_test_male <- x_test_male[which(x_test_male$Year > last_observed_year),]

                # Recursive prediction with respect to the gender.
                recursive_pred_female <- recursive_prediction(last_observed_year, x_test_female, "Female", countries, country, timesteps, age_range, model, number_of_projected_years) #, x_min, x_max)
                recursive_pred_male <- recursive_prediction(last_observed_year, x_test_male, "Male", countries, country, timesteps, age_range, model, number_of_projected_years) #, x_min, x_max)
                
                # Store projections in list.
                country_projections[[list_index]] <- list("Country" = country, "Female_Pred" = recursive_pred_female[[1]],
                                                          "Male_Pred" = recursive_pred_male[[1]])
                
                list_index <- list_index + 1
        }

        return(country_projections)

}