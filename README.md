# THESIS (2019)

**Main purpose of the tool:** Testing a (pseudo-)anonym dataset whether it hides privacy correctly, or leaks information about individuals.

Anonimity can be tested by three different metrics:
 - `kld`: Kullback-Leibler Divergence
 - `emd`: Earth Mover's Distance
 - `wmd`: Word Mover's Distance

Running the algorithm:
- Edit `src/properties.py` to declare the dataset (see below) and the number of records to test
  -  `CABS`    - location data
  -   `BMS`    - online commercial data
  - `MSNBC`    - web browsing data
- `$ python src/preprocess.py`: preprocessing phase of the dataset records
- `$ python src/methods/kld.py` | `emd.py` | `wmd.py`
- `*raw` files contain the results of the test
