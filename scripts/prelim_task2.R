# preliminary task 2

# load libraries
library(tidyverse)
library(tidymodels)
library(tidytext)
library(stopwords)
library(textstem)
library(Matrix)
library(irlba)
library(Metrics)
tidymodels_prefer()

<<<<<<< HEAD
# secondary tokenization of data to obtain bigrams with stop words
load('claims-clean-example.Rdata')
stoken <- claims_clean %>%
  mutate(text_clean = str_trim(text_clean)) %>%
  filter(str_length(text_clean) > 5) %>%
  unnest_tokens(output = 'token',
                input = text_clean,
                token = 'ngrams',
                n = 2) %>%
  group_by(.id, bclass) %>%
  count(token) %>%
  bind_tf_idf(term = token,
              document = .id,
              n = n) %>%
  pivot_wider(id_cols = c(.id, bclass),
              names_from = token,
              values_from = tf_idf,
              values_fill = 0) %>%
  ungroup()
=======
# retrieve pre-processed data for bigram PCA and logistic regression
load('data/claims-clean-bigram.RData')
load('data/claims-clean-singular.RData')
head(tokens_clean_bigram)
head(tokens_clean_singular)
#### PCA tokenized data then fit logistic regression ####
numeric_bigram_token <- tokens_clean_bigram |> 
  select(-.id, -bclass) |> 
  as.matrix()
>>>>>>> Reese's-Branch

# convert bigram to a sparse matrix to save computation
sparse_bigram <- as(numeric_bigram_token, "dgCMatrix")
pca_sparse <- irlba(sparse_bigram, nv = 10, center = T) # running PCA

# Extract principal components
pc_data <- as.data.frame(pca_sparse$u)
pc_data$bclass <- tokens_clean_bigram$bclass 

head(pc_data)

<<<<<<< HEAD
##### fit logistic pcr model to tokenized data (bigrams) #####
# Preprocess data
recipe <- recipe(bclass ~ ., data = stoken_train) |> 
  step_zv(all_predictors()) |>  # Remove zero-variance predictors
  step_normalize(all_numeric_predictors()) |>  # Normalize remaining predictors
  step_pca(all_numeric_predictors(), num_comp = 10)
=======
# Create a logistic regression model for bigrams only
logistic_bigram_pca <- glm(bclass ~., data = pc_data, family = binomial)
>>>>>>> Reese's-Branch

# Predict log-odds from bigram model
log_odds <- predict(logistic_bigram_pca, type = "link")

# Combine predicted log-odds with bigram PCs
pca_logit_combo <- pc_data
pca_logit_combo$log_odds <- log_odds

<<<<<<< HEAD
# train model on pca data first
pca_fit <- pca_workflow |> 
  fit(data = stoken_train)

# Predict log-odds ratios from the trained pca model
pca_pred_log_odds <- predict(pca_fit, stoken_train, type = "link") |> # type = "link" returns log-odds ratios
  bind_cols(stoken_train)

# Combine log-odds ratios with principal components of bigrams to train second model
combined_data <- pca_pred_log_odds |> 
  select(.id, bclass, .pred_link) |> 
  left_join(stoken_train, by = c(".id", "bclass")) |> 
  select(-bclass.y) |> 
  rename(bclass = bclass.x)

# Define a second logistic regression model
second_logistic_model <- logistic_reg(mode = "classification") |> 
  set_engine("glmnet")

# Create a new workflow for the second model
second_workflow <- workflow() |> 
  add_formula(bclass ~ .pred_link + PC1 + PC2 + PC3) |> # Can adjust PCs as needed
  add_model(second_logistic_model)

# Fit the second model
second_fit <- second_workflow |> 
  fit(data = combined_data)

# Evaluate the second model on testing data
second_pred <- predict(second_fit, stoken_test, type = "prob") |> 
  bind_cols(stoken_test)

# Metrics for evaluation
metrics <- metric_set(roc_auc, accuracy, precision, sensitivity, sensitivity)
second_eval_metrics <- metrics(second_pred, truth = bclass, estimate = .pred_class)

# Print metrics for the second model
print(second_eval_metrics)
=======
# Fit a second logistic regression model
logistic_combined <- glm(bclass ~ ., data = pca_logit_combo, family = binomial)

# Predict probabilities for test set
pc_predictions <- predict(logistic_bigram_pca, newdata = claims_test, type = "response")
combined_predictions <- predict(logistic_combined, newdata = claims_test, type = "response")

# Evaluate accuracy

bigram_accuracy <- accuracy_vec(truth = claims_test$bclass, estimate = pc_predictions)
combined_accuracy <- accuracy_vec(truth = claims_test$bclass, estimate = combined_predictions)

# Output results
cat("Accuracy with Bigram PCA:", bigram_accuracy, "\n")
cat("Accuracy with Combined Model:", combined_ac
>>>>>>> Reese's-Branch
