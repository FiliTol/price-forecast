# Price prediction for majori cities AirBnB listings

## Normalization, transformation, grouping

- [X] Transform other not-normally distributed variables in order to make them normally distributed. Use techniques like [`power_transform`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.power_transform.html).
- [X] After transformations set the normalization parameter to true in order to scale them immediately with the sklearn `power_transform` transformer.
- [ ] Change `amenities` computation by considering a bucket of fewer amenities and by keeping them as categories. Then the categories can be listed in the multi selection menu
- 