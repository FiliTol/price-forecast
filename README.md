# Price prediction for Venice-based AirBnB listings

## Visualization

### Feature engineering
- [X] `first_review` to `last_review` as `reviewed_time_span` in days
- [X] `host_listings_count` as a % of `host_total_listings_count`
- [X] Manage `neighbourhoods_cleansed` as a OHE of most frequent categories
- Distance between host home and listing location
- Distance between listing and relevant locations in town
- `host_since` encoded as *days of activity until period end (end of dataset scraping)*
- Sentiment of `neighborhood_overview` (investigate best sentiment technique for descriptions of appartments)
- Sentiment of `host_about` (investigate best sentiment technique for description of people)
- `host_id` as categorial (*with Count and Frequency Encoding from feature_engine package*) and **drop** `id` 
- `host_response_time` as ordinal variable
- string manipulation for `host_response_rate` and `host_acceptance_rate`
- [X] `host_is_superhost` as binary categorial
- `host_verifications` as encoded in previous script
- [X] `host_has_profile_pic` as binary
- [X] `host_identity_verified` as binary
- keep `room_type` instead of `property_type` and make `room_type` a categorial with OHE
- `accomodates` used with `baths`, `beds` to compute the rate of beds and bathrooms for every person
- `price` with string manipulation
- `minimum_nighs_avg_ntm` as float
- `maximum_nights_avg_ntm` as float
- [X] `has_availability` as binary
- all the `has_availability_NUMBER` as a % of the NUMBER of the feature
- `number_of_reviews` as an integer
- `review_scores_rating` as float
- all the reviews scores as float
- remove `calculated_host_listings_count` and keep the other three BUT **set them as % of the total host listings**
- `reviews_per_month` as float
- `longitude` and `latitude` standardization (because the values are both negatives and positives)

### Transform feature datatypes

In this section we execute the feature engineering without dealing with null values.
We do it because once the types are cleaned, we want to plot a bit the data and explore it to see what is going on
with NAs, frequency distribution, numeric distributions etc.
In order to do so, we need to:
1. Generalise the pipeline, because we would like to apply this script also to other similar dataset
2. Return exceptions for NAs, to carry them on to the data exploration section

> ***NOTE*** that the `feature-engine` library enables us to split the dataset into train and test just after the data type and feature engineering. This because the library contains some functions for [preprocessing](https://feature-engine.trainindata.com/en/latest/user_guide/preprocessing/index.html) that can deal with removed rows and features afterwards

- [Useful library for feature engineering](https://feature-engine.trainindata.com/en/latest/quickstart/index.html)

## Split features into groups based on the data type

- Split features for data types (***remember to insert the case where the columns with more than 50% NaN are not included in the splitting at all***)
    - Then the pipeline is build to transform the data types
    - Based on the previous splitting, apply Imputation methods to all the features. This is done because we don't know if other datasets will have the same null values ripartition
    - At this point we need to **drop** the columns not included in the splitting of data types. This because the columns not included will be the ones with a lot of NAs from the start (more than 50%)

> *Eventually we could compare the result of this approach with the result of a parallel approach whereby no columns are dropped and the NaNs are all Imputed. Then see how the two models perform*

