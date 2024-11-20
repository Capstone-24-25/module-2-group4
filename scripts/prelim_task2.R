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
load('data/claims-clean-singular.RData')
head(tokens_clean_bigram)
head(tokens_clean_singular)
#### PCA tokenized data then fit logistic regression ####
numeric_bigram_token <- tokens_clean_bigram |> 
  select(-.id, -bclass) |> 
  as.matrix()

# convert bigram to a sparse matrix to save computation
sparse_bigram <- as(numeric_bigram_token, "dgCMatrix")
pca_sparse <- irlba(sparse_bigram, nv = 10, center = T) # running PCA

# Extract principal components
pc_data <- as.data.frame(pca_sparse$u)
pc_data$bclass <- tokens_clean_bigram$bclass 

head(pc_data)

# Create a logistic regression model for bigrams only
logistic_bigram_pca <- glm(bclass ~., data = pc_data, family = binomial)

# Predict log-odds from bigram model
log_odds <- predict(logistic_bigram_pca, type = "link")

# Combine predicted log-odds with bigram PCs
pca_logit_combo <- pc_data
pca_logit_combo$log_odds <- log_odds

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