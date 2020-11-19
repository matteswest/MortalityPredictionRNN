# Functions for data preprocessing

# function that outputs training data set ( x_(t,x), Y_(t,x) )
data_preprocessing <- function(data.raw, gender, country, timesteps, feature_dimension, last_observed_years = 1999){ 

        mort_rates <- data.raw[which((data.raw$Gender == gender) & (data.raw$Country == country)), c("Year", "Age", "log_mortality")] 
        mort_rates <- dcast(mort_rates, Year ~ Age, value.var = "log_mortality")
        # selecting data
        train.rates <- as.matrix(mort_rates[which(mort_rates$Year <= last_observed_years),])
        # adding padding at the border
        (delta0 <- (feature_dimension-1)/2)
        if (delta0>0){
                for (i in 1:delta0){
                        train.rates <- as.matrix(cbind(train.rates[,1], train.rates[,2], train.rates[,-1], train.rates[,ncol(train.rates)]))
                }
        }
        train.rates <- train.rates[,-1]
        (t1 <- nrow(train.rates)-(timesteps-1)-1)
        (a1 <- ncol(train.rates)-(feature_dimension-1)) 
        (n.train <- t1 * a1) # number of training samples
        xt.train <- array(NA, c(n.train, timesteps, feature_dimension))
        YT.train <- array(NA, c(n.train))
        for (t0 in (1:t1)){
                for (a0 in (1:a1)){
                        xt.train[(t0-1)*a1+a0,,] <- train.rates[t0:(t0 + timesteps - 1), a0:(a0+feature_dimension-1)]
                        YT.train[(t0-1)*a1+a0] <- train.rates[t0 + timesteps, a0+delta0]
                }
        }
        list(xt.train, YT.train)

}



recursive_prediction <- function(last_observed_years, subdata, gender, country, timesteps, feature_dimension, model, x_min, x_max){

        yearly_mse <- array(NA, c(2016 - last_observed_years))

        for (current_year in ((last_observed_years+1):2016)){
                # Select only the necessary for the current year.
                data_current_year <- data_preprocessing(subdata[which(subdata$Year >= (current_year - timesteps)),], gender, country, timesteps, feature_dimension, current_year)

                # MinMaxScaler (with minimum and maximum from above)
                x_test <- array(2*(data_current_year[[1]]-x_min)/(x_min-x_max)-1, dim(data_current_year[[1]]))
                #x_test <- array(data_current_year[[1]], dim(data_current_year[[1]]))

                if (gender == "Female")
                        gender_index <- 0
                else
                        gender_index <- 1

                x_test <- list(x_test, rep(gender_index, dim(x_test)[1]))
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
