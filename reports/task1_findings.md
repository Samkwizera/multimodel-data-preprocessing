# Task 1 - Data Merge, EDA and Product Recommendation Model

Samuel Kwizera Ihimbazwe

## Approach

The task was to merge `customer_social_profiles` and `customer_transactions` into one
dataset and use it to predict the product category a customer buys. I did the merge and
the cleaning in `src/preprocessing.py`, the exploration in `notebooks/eda.ipynb`, and the
model in `src/product_model.py`. Everything below is reproducible by running those two
scripts.

## The join key

The two tables do not share a key. Social profiles identify a customer as
`customer_id_new` (`A178`, a string), transactions as `customer_id_legacy` (`178`, an
integer). Removing the `A` puts both onto the same `100-199` range, and since no mapping
table is provided and neither file holds any other candidate key, that is the only join
the data allows.

I want to be clear about how strong that assumption is, because it is the assumption the
whole task rests on:

- social profiles hold 84 unique ids
- transactions hold 75 unique ids
- after stripping the `A`, 61 of them overlap
- but two unrelated id sets of 84 and 75 drawn from a 100-value range would be expected
  to overlap on about 63 by chance alone

So the observed overlap is essentially what you would get if the two id sets had nothing
to do with each other. The id values give no evidence that `A178` and `178` are the same
person. I joined on it anyway because it is the only option available, but it is a stated
assumption rather than a verified fact, and the rest of the analysis inherits it.

## Why the merge is not a direct join

Both tables have several rows per customer. Social profiles hold 155 rows across 84
customers because a customer can appear on more than one platform, and transactions hold
150 rows across 75 customers because a customer can buy more than once.

Joining them directly multiplies profiles against transactions and produces rows that
never happened. In this data a direct join turns 150 transactions into 219 rows, with 102
transaction ids appearing more than once. Every duplicated transaction would then act as
extra training weight for a purchase that only occurred once.

I avoided this by collapsing the social table to one row per customer first (mean
engagement, mean purchase interest, modal platform, modal sentiment, and a count of
distinct platforms) and only then joining onto transactions. The result stays at one row
per real transaction: 150 rows in, 150 rows out, every `transaction_id` still unique.
`src/preprocessing.py` asserts both of those conditions on every run rather than leaving
them to be checked by eye.

## Cleaning

- **Duplicates.** The social table had 5 fully duplicated rows. These were dropped before
  aggregating, otherwise they would have pulled the per-customer means toward whichever
  profile happened to be repeated. The transaction table had none.
- **Types.** `purchase_date` was parsed to a datetime. The social id was cast to an
  integer after stripping the prefix so it could match the transaction key.
- **Nulls.** `customer_rating` was missing on 10 transactions and was filled with the
  median, with `rating_missing` recording where that happened.
- **Unmatched customers.** I used a left join, which keeps all 150 transactions. 117 of
  them (78%) matched a social profile and 33 did not. Those 33 are customers with no
  social profile at all rather than lost values, so filling their social columns with a
  median would quietly claim they had an average engagement score. I filled them so the
  data is model-ready but added `has_social_profile` so the model can tell a filled value
  from a measured one.

## Feature engineering

On top of the merged columns I added `purchase_month` and `purchase_dayofweek` from the
date, and per-customer spend history: `customer_txn_count`, `customer_avg_amount`, and
`amount_vs_customer_avg`. I deliberately did not build a "customer's most common product
category" feature, which would have handed the model its own target.

The final dataset is 150 rows and 18 columns with no nulls.

## What the EDA found

The target is close to balanced across five categories, so accuracy is a fair metric and
the majority class sits at 23.3%. That is the number any model has to beat.

The important finding is that nothing predicts the target. I tested every numeric feature
against `product_category` with a one-way ANOVA and every categorical one with a
chi-square test:

| feature | test | p |
|---|---|---|
| purchase_amount | ANOVA | 0.303 |
| customer_rating | ANOVA | 0.636 |
| engagement_score | ANOVA | 0.592 |
| purchase_interest_score | ANOVA | 0.076 |
| customer_avg_amount | ANOVA | 0.489 |
| amount_vs_customer_avg | ANOVA | 0.766 |
| social_media_platform | chi-square | 0.548 |
| review_sentiment | chi-square | 0.969 |
| has_social_profile | chi-square | 0.926 |

Every one is non-significant. The boxplots of each feature against product category sit
at the same level across all five classes, and the correlation heatmap shows nothing
except the relationships I built myself during feature engineering.

So `product_category` is close to independent of everything available to predict it. This
is a property of the dataset, not of the merge or the features, and no additional feature
engineering would change it.

## Model results

I compared a majority-class baseline, logistic regression, and a random forest.

The split needs care. `customer_avg_amount` and `customer_txn_count` are built from every
transaction a customer made, so if a customer's rows land on both sides of a random split,
the model can see their other purchases through those features. I split on the customer
rather than the row (`GroupShuffleSplit`, and `GroupKFold` for cross validation) so each
customer sits wholly on one side. The split reports 0 shared customers between train and
test.

Grouped 5-fold accuracy for the random forest is **0.220 +/- 0.045**, against a majority
class of **0.233**. The model is no better than always guessing the most common category,
and slightly worse.

Log loss says the same thing from another direction. The baseline, which only knows the
class frequencies, scores 1.671. Logistic regression scores 2.125 and the random forest
1.853. Both real models are worse calibrated than a baseline that knows nothing, which is
what it looks like when a model fits noise and is confidently wrong.

The clearest evidence came from fixing the split. Before I grouped by customer, the random
forest reached 0.316 on the holdout and 0.260 under plain 5-fold, which looks like a
modest lift over the 0.233 baseline. Once customers were held out properly, that lift
disappeared. It was never signal. The model was recognising customers it had already seen
through their spend aggregates.

## Conclusion

Four independent lines of evidence agree: the significance tests, the grouped cross
validation sitting below baseline, the log loss being worse than a frequency-only
baseline, and the apparent lift vanishing once the leakage was removed. The last one
matters most because it explains the other three rather than just repeating them.

A model on this data should score near 20-23%, and mine does. I would rather report a
model that performs at baseline for a reason I can demonstrate than one that looks strong
because the target leaked into the features or because it was scored on rows it trained
on.

The merge, the cleaning and the pipeline are sound, and the same pipeline would find real
structure if the data contained any. The limit here is the dataset.
