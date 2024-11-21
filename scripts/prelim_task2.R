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

# retrieve pre-processed data for bigram PCA and logistic regression
load('data/claims-clean-bigram.RData')
head(tokens_clean_bigram)

#### PCA tokenized data then fit logistic regression ####
numeric_bigram_token <- tokens_clean_bigram |> 
  select(-.id, -bclass) |> 
  as.matrix()

# convert bigram to a sparse matrix to save computation
sparse_bigram <- as(numeric_bigram_token, "dgCMatrix")
pca_sparse <- irlba(sparse_bigram, nv = 10) # running PCA

# Extract principal components
pc_data <- as.data.frame(pca_sparse$u) # $u for the u in SVD which is the PC's

# Ensure the response variable is a factor for classification
pc_data$bclass <- as.factor(pc_data$bclass)

# Create a logistic regression model for bigrams only
logistic_pca <- glm(bclass ~ ., data = pc_data, family = binomial)

# Predict probabilities from bigram model
log_odds <- predict(logistic_pca, type = "link") # ensure log odds are used

# Combine predicted probabilities with bigram PCs
pca_logit_combo <- pc_data |>
  bind_cols(log_odds)

# Ensure the response variable is a factor for classification in combined data
pca_logit_combo$bclass <- as.factor(pca_logit_combo$bclass)

# Define a second logistic regression model
second_logistic_model <- logistic_reg(mode = "classification") |> 
  set_engine("glmnet")

# Evaluate the second model on testing data
second_pred <- predict(second_logistic_model, pca_logit_combo, type = "prob") |> 
  bind_cols(pca_logit_combo)

# Metrics for evaluation
metrics <- metric_set(roc_auc, accuracy, precision, sensitivity, specificity)
second_eval_metrics <- metrics(second_pred, truth = bclass, estimate = .pred_class)

# Print metrics for the second model
print(second_eval_metrics)

# Predict probabilities for test set
pc_predictions <- predict(logistic_bigram_pca, newdata = claims_test, type = "response")
combined_predictions <- predict(logistic_combined, newdata = claims_test, type = "response")

# Evaluate accuracy
bigram_accuracy <- accuracy_vec(truth = claims_test$bclass, estimate = pc_predictions)
combined_accuracy <- accuracy_vec(truth = claims_test$bclass, estimate = combined_predictions)

# Output results
cat("Accuracy with Bigram PCA:", bigram_accuracy, "\n")
cat("Accuracy with Combined Model:", combined_accuracy, "\n")
