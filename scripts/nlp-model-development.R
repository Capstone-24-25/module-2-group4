## PREPROCESSING
#################

# can comment entire section out if no changes to preprocessing.R
#source('scripts/preprocessing.R')

# load raw data
#load('data/claims-raw.RData')

# preprocess (will take a minute or two)
#claims_clean <- claims_raw %>%
#  parse_data()

# export
#save(claims_clean, file = 'data/claims-clean-example.RData')

## MODEL TRAINING (NN)
######################
library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)

# load cleaned data
load('data/claims-clean-example.RData')

# partition into training and testing sets
set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)
# training set
train_text <- training(partitions) %>%
  pull(text_clean)
train_labels <- training(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1

# testing set
test_text <- testing(partitions) |> 
  pull(text_clean)
test_labels <- testing(partitions) |> 
  pull(bclass) |> 
  as.numeric() -1

# If having library conflicts
#install.packages("keras", type = "source")
#library(keras)
#install_keras()

# create a preprocessing layer
preprocess_layer <- layer_text_vectorization(
  standardize = NULL,
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = 10000,
  output_mode = 'tf_idf'
)
preprocess_layer %>% adapt(train_text)
# Preprocess the training and testing data
train_text_preprocessed <- preprocess_layer(train_text) %>% as.array()
test_text_preprocessed <- preprocess_layer(test_text) %>% as.array()

# define NN architecture
model <- keras_model_sequential() %>%
  preprocess_layer() %>%
  layer_dropout(0.2) %>% # prevent overfitting
  layer_dense(units = 100, activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

summary(model)

# configure for training
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

# train
history <- model %>%
  fit(train_text, 
      train_labels,
      validation_split = 0.3,
      epochs = 10)

## CHECK TEST SET ACCURACY HERE
test_pred <- model %>% predict()
test_accuracy <- mean((test_pred > 0.5) == test_labels)
cat("Test Accuracy: ", test_accuracy, "\n")
# compare to true labels

# save the entire model as a SavedModel
save_model_tf(model, "results/example-model")
