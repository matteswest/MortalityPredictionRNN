shuffle_data <- function(x_train, y_train) {

        sample_size <- length(y_train)

        new_x_train <- array(NA, dim = dim(x_train[[1]]))
        new_gender_indicator <- array(NA, dim = sample_size)
        new_country_indicator <- array(NA, dim = sample_size)
        new_y_train <- array(NA, dim = sample_size)

        index_list <- c(1:sample_size)
        # Shuffle index list.
        set.seed(10)
        index_list <- sample(index_list)

        for (index in 1:sample_size) {
                new_x_train[index, , ] <- x_train[[1]][index_list[index], , ]
                new_gender_indicator[index] <- x_train[[2]][index_list[index]]
                new_country_indicator[index] <- x_train[[2]][index_list[index]]
                new_y_train[index] <- y_train[index_list[index]]
        }

        list(list(new_x_train, new_gender_indicator, new_country_indicator), new_y_train)

}



get_country_index <- function(country, countries) {

        country_index <- 0
        for (current_country in countries) {
                if (country == current_country) {
                        return(country_index)
                        break
                }
                country_index <- country_index + 1
        }

}



plot_loss <- function(model_name, val_loss, loss) {

        plot(val_loss, type = "l", lwd = 2, col = "red", main = paste(model_name), xlab = "epochs", ylab = "MSE", log = "y")
        lines(loss, type = "l", col = "blue", lwd = 2)
        legend(x = "bottomleft", legend = c("validation loss", "loss"), lwd = c(2, 2), col = c("red", "blue"))

}