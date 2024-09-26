# ----------------------------------------------------------------------
# Vegetation Response Prediction using Rainfall and NDVI Data
# Capstone Project for HarvardX Professional Data Science Program
# Author: Benny Istanto (bennyistanto@gmail.com)
# Date: 26 September 2024
# 
# Description:
# This R script is part of the capstone project aimed at predicting vegetation 
# health (NDVI) based on rainfall data using machine learning models. The key 
# models implemented include:
# - Linear Regression
# - Random Forest Regression
# - ARIMAX (Autoregressive Integrated Moving Average with Exogenous Variables)
# The project also includes visualizations for NDVI difference and rainfall 
# lag influence on vegetation health over the study period from 2003-2023 in 
# the Indramayu region.
#
# Dataset Sources:
# - Rainfall data from CHIRPS v2
# - NDVI data from MODIS (Aqua and Terra satellites)
# The analysis focuses on dekadal (10-day interval) time series data.
# 
# Libraries:
# - dplyr, ggplot2, lubridate, randomForest, forecast, Metrics, RColorBrewer
# 
# The script covers the following steps:
# 1. Loading necessary libraries and datasets
# 2. Preprocessing data (handling missing values and scaling)
# 3. Feature engineering (creating lag features and adding seasonal information)
# 4. Exploratory data analysis (correlation analysis, trend visualization)
# 5. Predictive modeling (Linear Regression, Random Forest, ARIMAX)
# 6. Model evaluation (RMSE and MAE)
# 7. Visualizations (heatmaps and line plots)
#
# Input data: Rainfall and NDVI datasets from WFP at ADM2 level for the years 2003-2023
# ------------------------------------------------------------

# Load necessary libraries
library(dplyr)
library(tidyr)
library(readr)
library(zoo)  # For interpolation
library(caret)  # For scaling and normalization
library(randomForest)  # For Random Forest model
library(forecast)  # For ARIMAX model
library(ggplot2)  # For visualization
library(Metrics)  # For RMSE and MAE calculations
library(RColorBrewer)  # For heatmap color schemes

# ------------------------------------------------------------
# 1. Loading and Filtering Data
# ------------------------------------------------------------
# Define local file paths for the input data
rainfall_file <- "idn-rainfall-adm2-full.csv"
ndvi_file <- "idn-ndvi-adm2-full.csv"

# URLs for downloading data (optional if not already downloaded)
rainfall_data_url <- "https://data.humdata.org/dataset/e7b6ce3e-5a35-4c12-9ee9-76153da18bf3/resource/302565e6-3c7e-40e0-860d-9f08d83084b1/download/idn-rainfall-adm2-full.csv"
ndvi_data_url <- "https://data.humdata.org/dataset/f3234974-3ca9-4a10-a97a-674705eaeea7/resource/cb6098b6-faa8-48ca-96bf-b0cca6fac148/download/idn-ndvi-adm2-full.csv"

# Download and load the datasets if not already downloaded
if (!file.exists(rainfall_file)) {
  download.file(rainfall_data_url, rainfall_file, quiet = TRUE)
}
if (!file.exists(ndvi_file)) {
  download.file(ndvi_data_url, ndvi_file, quiet = TRUE)
}

# Load the datasets
rainfall_data <- readr::read_csv(rainfall_file)
ndvi_data <- readr::read_csv(ndvi_file)

# ------------------------------------------------------------
# 2. Filtering Data for Indramayu and Date Range
# ------------------------------------------------------------
# Filter for Indramayu (ADM2_PCODE = "ID3212") and for the period 2003-2023
indramayu_rainfall <- rainfall_data %>%
  filter(ADM2_PCODE == "ID3212", date >= "2003-01-01", date <= "2023-12-31")

indramayu_ndvi <- ndvi_data %>%
  filter(ADM2_PCODE == "ID3212", date >= "2003-01-01", date <= "2023-12-31")

# Convert 'date' column to Date format
indramayu_rainfall$date <- as.Date(indramayu_rainfall$date)
indramayu_ndvi$date <- as.Date(indramayu_ndvi$date)

# ------------------------------------------------------------
# 3. Preprocessing Data
# ------------------------------------------------------------
# Handling missing values using linear interpolation for rainfall and NDVI
indramayu_rainfall <- indramayu_rainfall %>%
  arrange(date) %>%
  mutate(rfh = na.approx(rfh),
         r1h = na.approx(r1h),
         r3h = na.approx(r3h))

indramayu_ndvi <- indramayu_ndvi %>%
  arrange(date) %>%
  mutate(vim = na.approx(vim))

# Scaling the data using caret (standardizing features)
preProc <- preProcess(indramayu_rainfall[, c("rfh", "r1h", "r3h")], method = c("center", "scale"))
rainfall_scaled <- predict(preProc, indramayu_rainfall)

preProc_ndvi <- preProcess(indramayu_ndvi[, "vim"], method = c("center", "scale"))
ndvi_scaled <- predict(preProc_ndvi, indramayu_ndvi)

# ------------------------------------------------------------
# 4. Feature Engineering
# ------------------------------------------------------------
# Creating lag features for rainfall to capture the delayed effect on NDVI
rainfall_lags <- rainfall_scaled %>%
  arrange(date) %>%
  mutate(rainfall_lag_1 = lag(rfh, 1),
         rainfall_lag_2 = lag(rfh, 2),
         rainfall_lag_3 = lag(rfh, 3))

# Join the lagged rainfall data with the scaled NDVI data
combined_data <- left_join(ndvi_scaled, rainfall_lags, by = c("date", "ADM2_PCODE"))

# Add a seasonal feature to the combined dataset
combined_data <- combined_data %>%
  mutate(month = as.numeric(format(as.Date(date), "%m")),
         wet_season = ifelse(month %in% c(11, 12, 1, 2, 3, 4), 1, 0))

# ------------------------------------------------------------
# 5. Exploratory Data Analysis (EDA)
# ------------------------------------------------------------
# Visualize the trends in rainfall and NDVI over time (for Indramayu)
ggplot(combined_data, aes(x = date)) +
  geom_line(aes(y = vim, color = "NDVI")) +
  geom_line(aes(y = rfh * 0.01, color = "Rainfall (scaled)")) +  # Scaling rainfall for comparison
  scale_y_continuous(sec.axis = sec_axis(~.*100, name = "Rainfall (mm)")) +
  labs(title = "Rainfall and NDVI Trends in Indramayu (2003-2023)",
       x = "Date", y = "NDVI", color = "Legend") +
  theme_minimal()

# ------------------------------------------------------------
# 6. Predictive Modeling
# ------------------------------------------------------------
# Random Forest Model to predict NDVI using lagged rainfall variables
rf_model <- randomForest(vim ~ rainfall_lag_1 + rainfall_lag_2 + rainfall_lag_3, 
                         data = combined_data, ntree = 500)

# Generate predictions using the Random Forest model
combined_data$predicted_ndvi_rf <- predict(rf_model, combined_data)

# Linear Regression Model for comparison
linear_model <- lm(vim ~ rainfall_lag_1 + rainfall_lag_2 + rainfall_lag_3, data = combined_data)
combined_data$predicted_ndvi_lm <- predict(linear_model, combined_data)

# Assign combined_data to cleaned_data for further use
cleaned_data <- combined_data  # Alias for consistency in heatmap sections

# ------------------------------------------------------------
# 7. Model Evaluation (RMSE, MAE)
# ------------------------------------------------------------
# Define a function to calculate RMSE and MAE
evaluate_model <- function(actual, predicted) {
  valid <- complete.cases(actual, predicted)
  rmse_val <- rmse(actual[valid], predicted[valid])
  mae_val <- mae(actual[valid], predicted[valid])
  list(RMSE = rmse_val, MAE = mae_val)
}

# Evaluate Random Forest model performance
rf_eval <- evaluate_model(combined_data$vim, combined_data$predicted_ndvi_rf)

# Evaluate Linear Regression model performance
lm_eval <- evaluate_model(combined_data$vim, combined_data$predicted_ndvi_lm)

# Print model performance
print(data.frame(Model = c("Random Forest", "Linear Regression"),
                 RMSE = c(rf_eval$RMSE, lm_eval$RMSE),
                 MAE = c(rf_eval$MAE, lm_eval$MAE)))

# ------------------------------------------------------------
# 8. Visualizations (Prediction Errors and Rainfall Influence Heatmaps)
# ------------------------------------------------------------
# NDVI Difference Heatmap
cleaned_data$predicted_ndvi_rf <- predict(rf_model, cleaned_data)
ndvi_difference <- actual_ndvi - cleaned_data$predicted_ndvi_rf
difference_matrix <- matrix(NA, nrow = length(unique(cleaned_data$year)), ncol = length(unique(cleaned_data$dekad)))

for (i in 1:length(unique(cleaned_data$year))) {
  for (j in 1:length(unique(cleaned_data$dekad))) {
    subset_data <- cleaned_data[cleaned_data$year == unique(cleaned_data$year)[i] & cleaned_data$dekad == unique(cleaned_data$dekad)[j], ]
    if (nrow(subset_data) > 0) {
      difference_matrix[i, j] <- mean(subset_data$vim - subset_data$predicted_ndvi_rf, na.rm = TRUE)
    }
  }
}

# Replace NA values with zeros
difference_matrix[is.na(difference_matrix)] <- 0

# Plot NDVI Difference Heatmap
layout(matrix(c(1,2), nrow = 1), widths = c(4,1))
par(mar = c(6, 6, 4, 2))
image(1:length(unique(cleaned_data$dekad)), 1:length(unique(cleaned_data$year)), t(difference_matrix), col = brewer.pal(7, "RdYlBu"), axes = FALSE, xlab = "Dekad", ylab = "Year", main = "NDVI Difference\nActual vs Random Forest Prediction")
axis(1, at = 1:length(unique(cleaned_data$dekad)), labels = unique(cleaned_data$dekad), cex.axis = 0.9)
axis(2, at = 1:length(unique(cleaned_data$year)), labels = unique(cleaned_data$year), cex.axis = 0.9)

# Add a color legend
par(mar = c(6, 1, 4, 5))
legend_values <- round(seq(min(difference_matrix, na.rm = TRUE), max(difference_matrix, na.rm = TRUE), length.out = 7), 2)
image(1, seq_along(legend_values), t(matrix(legend_values)), col = brewer.pal(7, "RdYlBu"), axes = FALSE)
axis(4, at = seq_along(legend_values), labels = legend_values, las = 1, cex.axis = 0.8)
title("NDVI Diff.", line = 2.5, cex.main = 0.9)

# Rainfall Lag Influence Heatmap
cor_matrix <- matrix(NA, nrow = length(unique(cleaned_data$year)), ncol = length(c("rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3", "rainfall_lag_4", "rainfall_lag_5", "rainfall_lag_6")))

for (i in 1:length(unique(cleaned_data$year))) {
  for (j in 1:6) {
    year_data <- cleaned_data[cleaned_data$year == unique(cleaned_data$year)[i], ]
    if (nrow(year_data) > 0) {
      cor_matrix[i, j] <- cor(year_data[[paste0("rainfall_lag_", j)]], year_data$predicted_ndvi_rf, use = "complete.obs")
    }
  }
}

# Plot Rainfall Lag Influence Heatmap

# Set up the plot layout and margins
layout(matrix(c(1, 2), nrow = 1), widths = c(4, 1))  
par(mar = c(6, 6, 4, 2))  # Adjust margins to increase heatmap size

# Create the heatmap for rainfall influence on predicted NDVI
image(
  1:6,  # Number of rainfall lags
  1:length(unique(cleaned_data$year)),  # Number of years
  t(cor_matrix),  # Transpose matrix for correct orientation
  col = brewer.pal(7, "RdYlBu"), 
  axes = FALSE, 
  xlab = "Rainfall Lag", 
  ylab = "Year",
  main = "Rainfall Lag Influence\non Predicted NDVI"
)
axis(1, at = 1:6, labels = c("rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3", 
                             "rainfall_lag_4", "rainfall_lag_5", "rainfall_lag_6"), cex.axis = 0.9)
axis(2, at = 1:length(unique(cleaned_data$year)), labels = unique(cleaned_data$year), cex.axis = 0.9)

# Add a color legend with rounded values
par(mar = c(6, 1, 4, 5))  # Adjust margin for the legend plot
legend_values <- round(seq(min(cor_matrix, na.rm = TRUE), 
                           max(cor_matrix, na.rm = TRUE), 
                           length.out = 7), 2)
image(1, seq_along(legend_values), t(matrix(legend_values)), 
      col = brewer.pal(7, "RdYlBu"), axes = FALSE)
axis(4, at = seq_along(legend_values), labels = legend_values, las = 1, cex.axis = 0.8)
title("Correlation", line = 2.5, cex.main = 0.9)

# ----------------------------------------------------------------------
# End of R Code: Vegetation Response Prediction Using Rainfall and NDVI
# ----------------------------------------------------------------------
