# preliminary task 2

# load libraries
library(tidyverse)
library(tidymodels)
library(tidytext)
library(stopwords)
library(textstem)

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

# visualize the data
stoken %>% head()

# split data into training and testing
partitions <- stoken %>%
  initial_split(prop = 0.7, strata = bclass)

stoken_train <- training(partitions) |> mutate(bclass = as.factor(bclass))
stoken_test <- testing(partitions) |> mutate(bclass = as.factor(bclass))

##### fit logistic pcr model to tokenized data (bigrams) #####
# Preprocess data
recipe <- recipe(bclass ~ ., data = stoken_train) |> 
  step_zv(all_predictors()) |>  # Remove zero-variance predictors
  step_normalize(all_numeric_predictors()) |>  # Normalize remaining predictors
  step_pca(all_numeric_predictors(), num_comp = 10)

# define logistic pcr model
logistic_model <- logistic_reg(mode = "classification") |> 
  set_engine("glmnet")

# create a workflow
pca_workflow <- workflow() |> 
  add_recipe(recipe) |> 
  add_model(logistic_model)

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
