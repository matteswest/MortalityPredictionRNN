shuffle_data <- function(x_train, y_train) {

        sample_size <- length(y_train)

        new_x_train <- array(NA, dim = dim(x_train[[1]]))
        new_gender_indicator <- array(NA, dim = sample_size)
        new_country_indicator <- array(NA, dim = sample_size)
        new_y_train <- array(NA, dim = sample_size)

        index_list <- c(1:sample_size)
        # Shuffle index list.
        index_list <- sample(index_list)

        for (index in 1:sample_size) {
                new_x_train[index, , ] <- x_train[[1]][index_list[index], , ]
                new_gender_indicator[index] <- x_train[[2]][index_list[index]]
                new_country_indicator[index] <- x_train[[2]][index_list[index]]
                new_y_train[index] <- y_train[index_list[index]]
        }

        list(list(new_x_train, new_gender_indicator, new_country_indicator), new_y_train)

}