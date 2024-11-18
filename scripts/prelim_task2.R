# preliminary task 2

# load libraries
library(tidyverse)
library(tidymodels)
library(tidytext)
library(stopwords)
library(textstem)

# secondary tokenization of data to obtain bigrams
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
# preprocess data
recipe <- recipe(bclass ~. , data = stoken_train) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_pca(all_numeric(), num_comp = 10) # adjust num_comp

# define logistic pcr model
logistic_model <- logistic_reg(mode = "classification") |> 
  set_engine("glmnet")

# create a workflow
pca_workflow <- workflow() |> 
  add_recipe(recipe) |> 
  add_model(logistic_model)

# fitting the first model
pca_fit <- pca_workflow |> 
  fit(data = stoken_train)

# extract predicted log-odds-ratios
pca_pred <- predict(pca_fit, stoken_test, type = "link") |> 
  bind_cols(stoken_test)

# combine log-odds-ratios with principal components
combined_data <- pca_pred |> 
  select(.pred_link, starts_with("PC")) # assuming PCs are named like "PC1", "PC2"

# fit a second logistic regression model
second_logistic_model <- logistic_reg(mode = "classification") |> 
  set_engine("glmnet")

second_workflow <- workflow() |> 
  add_formula(bclass ~ .) |> 
  add_model(second_logistic_model)

second_fit <- second_workflow |> 
  fit(data = combined_data)

# metrics to use for evaluation
metrics <- metric_set(roc_auc, accuracy, precision, sensitivity)

# Ensure predictions are converted to class labels
pca_pred_class <- predict(pca_fit, stoken_test, type = "class") |> 
  bind_cols(stoken_test)

# metrics for the first model
eval_metrics <- metrics(pca_pred_class, truth = bclass, estimate = .pred_class)

# Ensure predictions are converted to class labels for the second model
second_pred_class <- predict(second_fit, combined_data, type = "class") |> 
  bind_cols(combined_data)

# metrics for the second model
second_eval_metrics <- metrics(second_pred_class, truth = bclass, estimate = .pred_class)

# Print metrics for the second model
print(second_eval_metrics)