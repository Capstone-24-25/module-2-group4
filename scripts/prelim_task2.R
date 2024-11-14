# preliminary task 2

# load libraries
library(tidytext)
library(stopwords)
library(textstem)

# secondary tokenization of data to obtain bigrams
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

stoken %>% head()

partitions <- stoken %>%
  initial_split(prop = 0.7)

# fit logistic pcr model to tokenized data
