#' MovieLens Data Processing and Modeling Script
#'  
#' This script is part of the HarvardX Professional Data Science Capstone Project.
#' It processes the MovieLens 10M dataset, performs exploratory data analysis, 
#' and builds multiple models for predicting movie ratings. 
#' The models include baseline approaches such as mean rating and movie/user effects, 
#' as well as advanced techniques like regularized models and Singular Value Decomposition (SVD). 
#' The script also evaluates model performance using RMSE (Root Mean Square Error), 
#' providing a comprehensive evaluation of the predictive accuracy of the models.
#'
#' Dataset:
#' MovieLens 10M Dataset: https://grouplens.org/datasets/movielens/10m/
#' 
#' The analysis covers:
#' 1. Data Preprocessing: Loading, cleaning, and feature engineering.
#' 2. Exploratory Data Analysis: Examining rating distributions, genre trends, and user behavior.
#' 3. Model Building: Implementing baseline, regularized, and matrix factorization (SVD) models.
#' 4. Model Evaluation: Using RMSE to compare model performance.
#'
#' Author:
#' Benny Istanto
#' bennyistanto@gmail.com
#' September 2024
#' ---

# Load necessary libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(Matrix)) install.packages("Matrix", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(recosystem)
library(Matrix)

#' @section Load and preprocess the MovieLens dataset:
#' Download the MovieLens 10M dataset, load the ratings and movie metadata, 
#' and join them into a single dataset for analysis.

dl <- "ml-10M100K.zip"
if(!file.exists(dl)) download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
movies_file <- "ml-10M100K/movies.dat"

# Unzip and load the data
if(!file.exists(ratings_file)) unzip(dl, ratings_file)
if(!file.exists(movies_file)) unzip(dl, movies_file)

# Load and process ratings data
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE), stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>% mutate(userId = as.integer(userId), movieId = as.integer(movieId), rating = as.numeric(rating), timestamp = as.integer(timestamp))

# Load and process movies data
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE), stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>% mutate(movieId = as.integer(movieId))

# Join ratings and movie metadata
movielens <- left_join(ratings, movies, by = "movieId")

#' @section Exploratory Data Analysis:
#' Display basic statistics about the dataset and check for missing values.
# Display summary statistics for ratings and dataset dimensions
summary(movielens$rating)
dim(movielens)
sum(is.na(movielens))  # Check for missing values

#' @section Feature Engineering:
#' Create user and movie average rating features, extract time features, 
#' and generate dummy variables for genres.

# Create user and movie average ratings
user_avg_rating <- movielens %>% group_by(userId) %>% summarize(user_avg_rating = mean(rating))
movie_avg_rating <- movielens %>% group_by(movieId) %>% summarize(movie_avg_rating = mean(rating))

# Extract year from the timestamp for time-based analysis
movielens <- movielens %>% mutate(date = as.POSIXct(timestamp, origin = "1970-01-01"), year = format(date, "%Y"))

# Generate dummy variables for movie genres
movielens <- movielens %>% separate_rows(genres, sep = "\\|") %>% mutate(value = 1) %>% spread(genres, value, fill = 0)

#' @section Scaling Ratings:
#' Normalize the rating column to have a mean of 0 and a standard deviation of 1.
# Standardizing ratings to prevent scale issues in models
movielens <- movielens %>% mutate(scaled_rating = scale(rating))

# Display the first few rows of the processed dataset
head(movielens)

#' @section Baseline Model - Mean Rating:
#' Build a baseline model that predicts the average rating across all movies, 
#' and calculate the RMSE for this model.
mean_rating <- mean(movielens$rating)  # Global mean rating
baseline_rmse <- rmse(movielens$rating, rep(mean_rating, nrow(movielens)))  # Baseline RMSE calculation

#' @section Movie Effect Model:
#' Adjust the mean rating by adding a movie-specific bias for each movie.
#' Calculate the RMSE for the movie effect model.
movie_avg <- movielens %>% group_by(movieId) %>% summarize(movie_bias = mean(rating - mean_rating))
movielens <- movielens %>% left_join(movie_avg, by = "movieId") %>% mutate(pred_movie_effect = mean_rating + movie_bias)
movie_effect_rmse <- rmse(movielens$rating, movielens$pred_movie_effect)

#' @section User Effect Model:
#' Further adjust the predictions by adding a user-specific bias.
#' Calculate the RMSE for the user effect model.
user_avg <- movielens %>% group_by(userId) %>% summarize(user_bias = mean(rating - (mean_rating + movie_bias)))
movielens <- movielens %>% left_join(user_avg, by = "userId") %>% mutate(pred_user_effect = mean_rating + movie_bias + user_bias)
user_effect_rmse <- rmse(movielens$rating, movielens$pred_user_effect)

#' @section Regularization Model:
#' Implement regularization to avoid overfitting by penalizing large movie and user biases.
#' Calculate the RMSE for the regularization model using the best lambda value.

# Grid search for the best lambda (regularization parameter)
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l) {
  # Movie bias with regularization
  movie_avg_reg <- movielens %>% group_by(movieId) %>% summarize(movie_bias = sum(rating - mean_rating) / (n() + l))
  # User bias with regularization
  user_avg_reg <- movielens %>% left_join(movie_avg_reg, by = "movieId") %>% group_by(userId) %>% summarize(user_bias = sum(rating - (mean_rating + movie_bias)) / (n() + l))
  # Predict using the regularized biases
  movielens_with_bias <- movielens %>% left_join(movie_avg_reg, by = "movieId") %>% left_join(user_avg_reg, by = "userId") %>% mutate(pred_regularized = mean_rating + movie_bias + user_bias)
  return(rmse(movielens$rating, movielens_with_bias$pred_regularized))
})

best_lambda <- lambdas[which.min(rmses)]  # Select the best lambda
# Recalculate regularized RMSE with the best lambda
movie_avg_reg <- movielens %>% group_by(movieId) %>% summarize(movie_bias = sum(rating - mean_rating) / (n() + best_lambda))
user_avg_reg <- movielens %>% left_join(movie_avg_reg, by = "movieId") %>% group_by(userId) %>% summarize(user_bias = sum(rating - (mean_rating + movie_bias)) / (n() + best_lambda))
movielens <- movielens %>% left_join(movie_avg_reg, by = "movieId") %>% left_join(user_avg_reg, by = "userId") %>% mutate(pred_regularized = mean_rating + movie_bias + user_bias)
regularized_rmse <- rmse(movielens$rating, movielens$pred_regularized)

#' @section SVD Model:
#' Use the Reco library to perform matrix factorization (SVD) on the MovieLens dataset, 
#' and calculate the RMSE for the SVD model.

# Preparing data for the SVD model
train_data <- movielens %>% select(userId, movieId, rating)
write.table(train_data, file = "train_svd.txt", sep = " ", row.names = FALSE, col.names = FALSE)
r <- Reco()

# Train the SVD model on the training data
r$train(data_file("train_svd.txt"))

# Predict ratings using the trained SVD model
predicted_svd <- r$predict(data_file("train_svd.txt"))
svd_rmse <- rmse(movielens$rating, predicted_svd)

#' @section Output RMSE Results:
#' Print the RMSE for each model to compare performance.
baseline_rmse
movie_effect_rmse
user_effect_rmse
regularized_rmse
svd_rmse
