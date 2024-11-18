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

# fitting the model
pca_fit <- pca_workflow |> 
  fit(data = stoken_train)

# evaluate the model on testing
pca_pred <- predict(pca_fit, stoken_test, type = "prob") |> 
  bind_cols(stoken_test)

# metrics to use for evaluation
metrics <- metric_set(roc_auc, accuracy, precision, sensitivity, sensitivity)
eval_metrics <- metrics(pca_pred, truth = bclass, estimate = .pred_class)

# Print metrics
print(eval_metrics)
