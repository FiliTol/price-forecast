# Price prediction for majori cities AirBnB listings

## Manipulation of multiple cities

- [ ] Add features regarding some statistics about the city
- [ ] During the ML pipeline, change the `latitude`, `longitude` into **polars coordinates** ([as suggested here](https://stackoverflow.com/questions/61572370/dealing-with-longitude-and-latitude-in-feature-engineering))
- [X] Keep the `property_type` of the listing and remap it into a smaller set of options (*already have it in draft notebook*)
- [X] Rework the `host_id` reasoning considering that hosts could have listings in multiple cities
- [X] Remove immediately text variables, also `host_about` and `neighborhood_overview`
- [X] **Remove the variables regarding the distance between the venetian monuments and the listing**
- [X] Analyse again the `bathrooms` and `bathrooms_text` variables at dispose to understand if there are a lot of bathrooms missing or not and eventually change the cleaning approach
- [X] Handle `amenities` by remapping them into smaller sets

## Normalization, transformation, grouping

- [ ] Group highly skewed data like `host_response_rate` and `host_acceptance_rate` as `100` vs `lower`.
- [ ] Transform other not-normally distributed variables in order to make them normally distributed. Use techniques like [`power_transform`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.power_transform.html).
- [ ] After transformations set the normalization parameter to true in order to scale them immediately with the sklearn `power_transform` transformer.
