# Import libraries
library(tidyverse)
library(data.table)
library(keras)

# import source codes
source("data_preparation.R")

# Set parameters.
T0 <- 10
tau0 <- 5
tau1 <- 20
tau2 <- 15
obsYear <- 2006
country <- "DEUT"

# Load data.
data <- fread("https://raw.githubusercontent.com/DeutscheAktuarvereinigung/Mortality_Modeling/master/mortality.csv")
# Convert gender and country to factor variables.
data$Gender <- as.factor(data$Gender)
data$Country <- as.factor(data$Country)
# Add column for mortality.
data$mortality <- exp(data$log_mortality)
# Filter relevant countries.
data <- data[which(data$Country %in% c("CHE", "DEUT", "DNK", "ESP", "FRATNP", "ITA", "JPN", "POL", "USA")),]


#### Both Genders
data_female <- data.preprocessing(data, "Female", country,T0 = 10, tau0 = 5, obsYears)
data_male <- data.preprocessing(data, "Male", country, T0 = 10, tau0 = 5, obsYears)

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

# MinMaxScaler data pre-processing
x.min <- min(x.train)
x.max <- max(x.train)
x.train <- list(array(2*(x.train-x.min)/(x.min-x.max)-1, dim(x.train)), gender.indicator)

# The mean of y.train will be used as starting value for the intercept weight as it leeds to 
# faster convergence
y0 <- mean(y.train)


# validation data pre-processing
data2.Female <- data[which((data$Year > (obsYear-10))&(Gender=="Female")),]
dataV.Female <- data2.Female
vali.Y.Female <- dataV.Female[which(dataV.Female$Year > obsYear),]
data2.Male <- data[which((data$Year > (obsYear-10))&(Gender=="Male")),]
dataV.Male <- data2.Male
vali.Y.Male <- dataV.Male[which(dataV.Male$Year > obsYear),]

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
        history <- model %>% fit(x = x.train, y = y.train, validation_split = 0.2, batch_size = 100, epochs = 150, verbose = 1)
        Sys.time() - current_time}
plot(history)

# calculating out-of-sample loss: LC is c(Female=0.6045, Male=1.8152)
# Female
pred.result <- recursive.prediction.Gender(obsYear, data2.Female, "Female", T0, tau0, x.min, x.max, model)
vali <- pred.result[[1]][which(data2.Female$Year > obsYear),]
round(10^4*mean((vali$mx-vali.Y.Female$mx)^2),4)
# Male
pred.result <- recursive.prediction.Gender(obsYear, data2.Male, "Male", T0, tau0, x.min, x.max, model)
vali <- pred.result[[1]][which(data2.Male$Year > obsYear),]
round(10^4*mean((vali$mx-vali.Y.Male$mx)^2),4)