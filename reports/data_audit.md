| step                      |   rows_before |   rows_after |   removed | rule                                                           |
|:--------------------------|--------------:|-------------:|----------:|:---------------------------------------------------------------|
| raw load                  |          9974 |         9974 |         0 | file sha256 bac4837af685                                       |
| exact duplicates          |          9974 |         7474 |      2500 | key = location, size, price, bedrooms, bathrooms               |
| price and size bounds     |          7474 |         7237 |       237 | price 150,000-7,000,000 SAR, size 100-1200 m2                  |
| price per m2 rule (train) |          5786 |         5770 |        16 | 1,000-15,000 SAR/m2                                            |
| price per m2 rule (test)  |          1451 |         1448 |         3 | 1,000-15,000 SAR/m2; removed rows kept for a sensitivity check |