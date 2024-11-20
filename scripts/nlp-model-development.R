## PREPROCESSING
#################

# can comment entire section out if no changes to preprocessing.R
source('scripts/preprocessing.R')
# load raw data
load('data/claims-raw.RData')
# preprocess (will take a minute or two)
claims_clean <- claims_raw %>%
  parse_data()
tokens_clean <- claims_clean %>% # singular tokenization
  nlp_fn()
tokens_clean_bigram <- claims_clean %>% # bigram tokenization
  nlp_fn_bigram()

# export
save(tokens_clean, file = 'data/claims-clean-singular.RData')
save(tokens_clean_bigram, file = 'data/claims-clean-bigram.RData')
save(claims_clean, file = 'data/claims-clean-example.RData')

## MODEL TRAINING (NN)
######################
library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)

# load cleaned data
load('data/claims-clean-example.RData')
head(claims_clean)
# partition into training and testing sets
set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>% pull(text_clean)
train_labels <- training(partitions) %>% pull(bclass) %>% as.numeric() - 1
test_text <- testing(partitions) %>% pull(text_clean)
test_labels <- testing(partitions) %>% pull(bclass) %>% as.numeric() - 1

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
  layer_dense(units = 128, activation = 'relu', input_shape = dim(train_text_preprocessed)[-1]) %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(0.2) %>% 
  layer_dense(units = 32, activation = 'relu') %>% 
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
callback <- callback_early_stopping(monitor="val_loss", patience = 4)
history <- model %>%
  fit(train_text_preprocessed, 
      train_labels,
      validation_split = 0.3,
      epochs = 10)
      callbacks = list(callback)) # prevent overfit

## CHECK TEST SET ACCURACY HERE
test_pred <- model %>% evaluate(test_text_preprocessed, test_labels)
cat("Test Loss:", test_pred[1], "Test Accuracy:", test_pred[2], "\n")

# save the entire model as a SavedModel
save_model_tf(model, "results/NN_claims_clean")
