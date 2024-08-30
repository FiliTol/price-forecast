# Price prediction for majori cities AirBnB listings

## Normalization, transformation, grouping

- [X] Transform other not-normally distributed variables in order to make them normally distributed. Use techniques like [`power_transform`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.power_transform.html).
- [X] After transformations set the normalization parameter to true in order to scale them immediately with the sklearn `power_transform` transformer.
- [ ] Change `amenities` computation by considering a bucket of fewer amenities and by keeping them as categories. Then the categories can be listed in the multi selection menu
- [ ] compute for every dataset of every city the distance BY FOOT from several standardized points of interest
  - City center (P.za San Marco)
  - Airport
  - Main train station
 - [Create path between geo objects](https://osmnx.readthedocs.io/en/stable/)
 - [Compute effective travel distance](https://lenkahas.com/post/pandana.html)
 - [Some city tourism statistics](https://ec.europa.eu/eurostat/statistics-explained/index.php?title=City_statistics_-_tourism)
 - 