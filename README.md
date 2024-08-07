# Price prediction for majori cities AirBnB listings

## Manipulation of multiple cities

- [ ] Add features regarding some statistics about the city
- [ ] During the ML pipeline, change the `latitude`, `longitude` into **polars coordinates** ([as suggested here](https://stackoverflow.com/questions/61572370/dealing-with-longitude-and-latitude-in-feature-engineering))
- [ ] Keep the `property_type` of the listing and remap it into a smaller set of options (*already have it in draft notebook*)
- [ ] Rework the `host_id` reasoning considering that hosts could have listings in multiple cities
- [ ] Remove immediately text variables, also `host_about` and `neighborhood_overview`
- [ ] **Remove the variables regarding the distance between the venetian monuments and the listing**
- [ ] Analyse again the `bathrooms` and `bathrooms_text` variables at dispose to understand if there are a lot of bathrooms missing or not and eventually change the cleaning approach
- [ ] Handle `amenities` by remapping them into smaller sets
- [ ] 
