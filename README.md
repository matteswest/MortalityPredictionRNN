# MortalityPredictionRNN

This repository contains R code to model and predict mortality rates in 9 different countries using the Human Mortality Database (HMD) dataset which is stored in the 'data' directive. 
The predictions are made using Recurrent Neural Networks based on R. Richman and M.V. Wüthrich's [code](https://github.com/JSchelldorfer/ActuarialDataScience/tree/master/6%20-%20Lee%20and%20Carter%20go%20Machine%20Learning%20Recurrent%20Neural%20Networks). Both LSTM and GRU architectures are possible. 

How to work with this repository:

There are two options: 

If you only want to create and fit a specific model with specific parameters you can use the script rnn_mortality.R and set the model parameters in
the head of the code. 

To perform a complete hyperparameter search in order to find the best performing models based on the Mean-Squared-Error using different combinations
of parameters do the following:

1. Run the grid_search script using your parameter options in the beginning of the code. Note that, for performance issues, the tuning_run function in the last line of the code
has the 'sample' parameter set to 0.01, so that it takes only a 1% sample of the possible parameter combinations. The script will then perform a grid search (by executing the hyperparamters.R script for each model) saving all models in a new subdirectory called 'grid_search_LSTM' or 'grid_search_GRU' respectively. 

2. To evaluate the results of the grid search you can use the evaluate_grid_search.R script. Here you can activate and deactivate different evaluation methods. You can also calculate the out of sample loss for an ensemble of models by setting the number_of_models variable to a number greater than 1. The code will then load the first <number_of_models> best models of the grid search and calculate the average out of sample loss over these models. 
