# Functions for data preprocessing

# function that outputs training data set ( x_(t,x), Y_(t,x) )
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


# Recursive Prediction for both genders (copied from 
# https://github.com/JSchelldorfer/ActuarialDataScience/blob/master/6%20-%20Lee%20and%20Carter%20go%20Machine%20Learning%20Recurrent%20Neural%20Networks/00_c%20package%20-%20data%20preparation%20RNNs.R)

recursive.prediction.Gender <- function(ObsYear, all_mort2, gender, T0, tau0, x.min, x.max, model.p){       
        single.years <- array(NA, c(2016-ObsYear))
        
        for (ObsYear1 in ((ObsYear+1):2016)){
                data2 <- data.preprocessing(all_mort2[which(all_mort2$Year >= (ObsYear1-10)),], gender, T0, tau0, ObsYear1)
                # MinMaxScaler (with minimum and maximum from above)
                x.vali <- array(2*(data2[[1]]-x.min)/(x.min-x.max)-1, dim(data2[[1]]))
                if (gender=="Female"){yy <- 0}else{yy <- 1}
                x.vali <- list(x.vali, rep(yy, dim(x.vali)[1]))
                y.vali <- -data2[[2]]
                Yhat.vali2 <- exp(-as.vector(model.p %>% predict(x.vali)))
                single.years[ObsYear1-ObsYear] <- round(10^4*mean((Yhat.vali2-exp(-y.vali))^2),4)
                predicted <- all_mort2[which(all_mort2$Year==ObsYear1),]
                keep <- all_mort2[which(all_mort2$Year!=ObsYear1),]
                predicted$logmx <- -as.vector(model %>% predict(x.vali))
                predicted$mx <- exp(predicted$logmx)
                all_mort2 <- rbind(keep,predicted)
                all_mort2 <- all_mort2[order(Gender, Year, Age),]
        }
        list(all_mort2, single.years)
}                  
