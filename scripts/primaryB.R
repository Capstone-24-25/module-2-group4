# primary B 
# predictive model for multi-class classification

# load libraries
library(randomForest)
library(tm)
library(dplyr)
library(tidytext)
library(quanteda)
library(tidyr)
library(keras)

# load data
load("data/claims-clean-example.RData")
load("data/claims-test.RData")

# factor labels to integers
train_labels <- to_categorical(as.numeric(claims_clean$mclass) - 1)

# text preprocessing
# preprocess training data (claims_clean)
claims_clean$text_clean <- claims_clean$text_tmp %>%
  gsub("<.*?>", " ", .) %>%        # remove HTML tags
  gsub("\\s+", " ", .) %>%         # remove extra whitespace
  trimws()                         # trim leading/trailing spaces

# preprocess test data (claims_test)
claims_test$text_clean <- claims_test$text_tmp %>%
  gsub("<.*?>", " ", .) %>%
  gsub("\\s+", " ", .) %>%
  trimws()

# labels to factor to numeric for the training dataset
claims_clean$mclass <- as.factor(claims_clean$mclass)
claims_clean$mclass <- as.numeric(claims_clean$mclass) - 1  # Zero-based indexing

train_labels_one_hot <- to_categorical(claims_clean$mclass)

# create and fit tokenizer
tokenizer <- text_tokenizer(num_words = 10000)
tokenizer %>% fit_text_tokenizer(claims_clean$text_clean)

vocab_size <- length(tokenizer$word_index) + 1

# tokenize train and test text
train_sequences <- texts_to_sequences(tokenizer, claims_clean$text_clean)
test_sequences <- texts_to_sequences(tokenizer, claims_test$text_clean)

# pad sequences to the same length
maxlen <- 200  # fixed for consistency
train_padded <- pad_sequences(train_sequences, maxlen = maxlen)
test_padded <- pad_sequences(test_sequences, maxlen = maxlen)

# model with L2 regularization to improve accuracy
model_lstm <- keras_model_sequential() %>%
  layer_embedding(input_dim = vocab_size, output_dim = 100, input_length = maxlen) %>%
  layer_lstm(units = 128, 
             kernel_regularizer = regularizer_l2(0.01),
             dropout = 0.2, 
             recurrent_dropout = 0.2) %>%
  layer_dense(units = 64, activation = 'relu',
              kernel_regularizer = regularizer_l2(0.01)) %>%  
  layer_dense(units = num_classes, activation = 'softmax')

# compile model
model_lstm %>% compile(
  loss = 'categorical_crossentropy', 
  optimizer = optimizer_adam(), 
  metrics = c('accuracy')
)


# fit model and capture history
history_lstm <- model_lstm %>% fit(
  train_padded, 
  train_labels_one_hot,  # One-hot encoded labels
  validation_split = 0.2, 
  epochs = 10,
  batch_size = 64,
  callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 3))
)

# make predictions on the test set
test_predictions <- model_lstm %>% predict(test_padded)

# convert predicted class probabilities to class labels
predicted_classes <- apply(test_predictions, 1, which.max) - 1  # Convert 1-based index to 0-based

# output data frame
pred_df <- data.frame(.id = claims_test$.id, mclass.pred = predicted_classes)
head(pred_df)

# predictions saved to CSV file
write.csv(pred_df, "predictions_lstm.csv", row.names = FALSE)

# saving model to file
model_lstm %>% save_model_tf("model_lstm.keras")

# model and predictions in one zip
zip("model_and_predictions.zip", c("model_lstm.keras", "predictions_lstm.csv"))

# load model from file
loaded_model <- load_model_tf("model_lstm.keras")
