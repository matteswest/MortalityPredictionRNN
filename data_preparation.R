# Functions for data preprocessing
# function that outputs training data set
data.preprocessing <- function(data.raw, gender, country, T0, tau0, ObsYear=1999){ 
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