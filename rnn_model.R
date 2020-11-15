# Import libraries
library(tidyverse)
library(data.table)
library(keras)

# Set parameters.
T0 <- 10
tau0 <- 5
tau1 <- 20
tau2 <- 15
obsYears <- 2006
country <- "DEUT"

# Load data.
data <- fread(file="mortality.csv")
# Convert gender and country to factor variables.
data$Gender <- as.factor(data$Gender)
data$Country <- as.factor(data$Country)
# Add column for mortality.
data <- data %>% mutate(mortality=exp(log_mortality))
# Filter relevant countries.
data <- dplyr::filter(data,Country%in%c("CHE", "DEUT", "DNK", "ESP", "FRATNP", "ITA", "JPN", "POL", "USA"))


# function that outputs training data set
data.preprocessing.RNNs <- function(data.raw, gender, country, T0, tau0, ObsYear=1999){ 
        mort_rates <- data.raw[which((data.raw$Gender == gender) & (data.raw$Country == country)), c("Year", "Age", "log_mortality")] 
        mort_rates <- dcast(mort_rates, Year ~ Age, value.var = "log_mortality")
        # selecting data
        train.rates <- as.matrix(mort_rates[which(mort_rates$Year <= ObsYear),])
        # adding padding at the border
        (delta0 <- (tau0-1)/2)
        if (delta0>0){
                for (i in 1:delta0){
                        train.rates <- as.matrix(cbind(train.rates[,1], train.rates[,2], train.rates[,-1], train.rates[,ncol(train.rates)]))
                }
        }
        train.rates <- train.rates[,-1]
        (t1 <- nrow(train.rates)-(T0-1)-1)
        (a1 <- ncol(train.rates)-(tau0-1)) 
        (n.train <- t1 * a1) # number of training samples
        xt.train <- array(NA, c(n.train, T0, tau0))
        YT.train <- array(NA, c(n.train))
        for (t0 in (1:t1)){
                for (a0 in (1:a1)){
                        xt.train[(t0-1)*a1+a0,,] <- train.rates[t0:(t0+T0-1), a0:(a0+tau0-1)]
                        YT.train[(t0-1)*a1+a0] <- train.rates[t0+T0, a0+delta0]
                }
        }
        list(xt.train, YT.train)
}

#### Both Genders
data_female <- data.preprocessing.RNNs(data, "Female", country,T0 = 10, tau0 = 5, obsYears)
data_male <- data.preprocessing.RNNs(data, "Male", country, T0 = 10, tau0 = 5, obsYears)

# check if dimensions of male and female data match
if ( (dim(data_female[[1]])[1] != dim(data_male[[1]])[1]) | (dim(data_female[[2]])[1] != dim(data_male[[2]])[1]) ) {
        stop("Shapes of female and male are not the same!")
}

# merge female and male data into one set
xx <- dim(data_female[[1]])[1]
x.train <- array(NA, dim=c(2*xx, dim(data_female[[1]])[c(2,3)]))
y.train <- array(NA, dim=c(2*xx))
gender.indicator <- rep(c(0,1), xx)
for (l in 1:xx){
        x.train[(l-1)*2+1,,] <- data_female[[1]][l,,]
        x.train[(l-1)*2+2,,] <- data_male[[1]][l,,]
        # Invert label sign.
        y.train[(l-1)*2+1] <- - data_female[[2]][l]
        y.train[(l-1)*2+2] <- - data_male[[2]][l]
}

# The mean of y.train will be used as starting value for the intercept weight as it leeds to 
# faster convergence
y0 <- mean(y.train)

# Define network. Here we use two LSTM layers.
Input <- layer_input(shape=c(T0,tau0), dtype='float32', name='Input') 
Output = Input %>%  
        layer_lstm(units = tau1, activation = 'tanh', recurrent_activation = 'tanh', 
                         return_sequences = TRUE, name = 'LSTM1') %>%
        layer_lstm(units = tau2, activation='tanh', recurrent_activation ='tanh', name='LSTM2') %>%           
        layer_dense(units = 1, activation = NULL, name = "Output",
                    weights = list(array(0,dim = c(tau2,1)), array(log(y0),dim = c(1))))  
model <- keras_model(inputs = list(Input), outputs = c(Output))

# Compile network.
model %>% compile(optimizer = "adam", loss = "mse")

#CBs <- callback_model_checkpoint(file.name, monitor = "val_loss", verbose = 0,  save_best_only = TRUE, save_weights_only = TRUE)
{current_time <- Sys.time()
        history <- model %>% fit(x = x.train, y = y.train, validation_split = 0.2, batch_size = 100, epochs = 25, verbose = 1)
        Sys.time() - current_time}
plot(history)
