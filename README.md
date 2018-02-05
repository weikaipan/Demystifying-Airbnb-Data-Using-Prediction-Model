#    Demystifying Airbnb Data Using Prediction Model

The analysis will identify opportunities and areas where properties can generate high gross margins by comparing the buying cost (annual mortgage) 
and expected annual revenue generated from both short term rentals on Airbnb and long-term rental price. The focus will be on visualizing up-and-
coming markets where the margins are high, as well as a comparison between long-term lease and short-term rentals. Along with these data, current 
regulations on Airbnb rentals in each city will also be taken into consideration. The project will use Hadoop and Pig to calculate aggregate statistics and 
discover which areas is recommended for long-term rental or short-term rental.

## Authors

* **Yi-Hsuan Fu**
* **Chun-Hao Yang**
* **Wei-Kai Pan**


### Prerequisites

```
Python 3.6.0
Java 1.8.0_72
Spark 2.2.0
```

### Running on [Dumbo](https://wikis.nyu.edu/display/NYUHPC/Clusters+-+Dumbo)

```
module load java/1.8.0_72
module load spark/2.2.0
module load anaconda3/4.3.1
PYTHONSTARTUP=analysis.py pyspark
```

## Data Source 
* [Inside Airbnb](http://insideairbnb.com/get-the-data.html), Listings of NYC, range from Jan. 2017 to Oct. 2017
* [NYPD Crime Statistics](http://www1.nyc.gov/site/nypd/stats/crime-statistics/crime-statistics-landing.page)
* [Zillow](https://www.zillow.com/home-values/)

## Analytic Process 

* [Data Flow Diagram](Data-Flow-Diagram.pdf)

### Data Cleaning

```
Hadoop and MapReduce
```

### Encoding Categorical Data


```
Spark MLlib - OneHotEncoder
```
### Correlation Analysis and Feature Selection 

```
Spark-ML-Statistics - Chi-Square Test
```
### Random Forest Regression and Gradient Boosted Regressor

```
Spark ML
```


## Built With

* [Apache Spark MLlib](https://spark.apache.org/mllib/) 
* [Apache Hadoop](http://hadoop.apache.org/)
* [Apache SparkSQL](https://spark.apache.org/sql/)
* [GEOJSON](http://geojson.org/)

## Result


## Future Work

## License

This project is licensed under the NYU License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Suzanne McIntosh
* [Inside Airbnb](http://insideairbnb.com/index.html)
* [NYPD Crime Statistics](http://www1.nyc.gov/site/nypd/stats/crime-statistics/crime-statistics-landing.page)

## References

* T. White. Hadoop: The Definitive Guide. O’Reilly Media Inc., Sebastopol, CA, May 2012.
* [Matei Zaharia, Mosharaf Chowdhury, Michael J. Franklin, Scott Shenker, Ion Stoica : Spark: Cluster Computing withWorking Sets](https://people.csail.mit.edu/matei/papers/2010/hotcloud_spark.pdf)
* Tita, G. E., Petras, T. L., & Greenbaum, R. T. (2006). Crime and Residential Choice: A Neighborhood Level Analysis of the Impact of Crime on Housing Prices. Journal of Quantitative Criminology,22(4), 299-317. doi:10.1007/s10940-006-9013-z
* [Chen, L., & Liang, W. (n.d.). Airbnb price prediction using Gradient boosting(Tech.).](https://cseweb.ucsd.edu/classes/wi17/cse258-a/reports/a043.pdf)
* Zhang, Z., Chen, R., Han, L., & Yang, L. (2017). Key Factors Affecting the Price of Airbnb Listings: A Geographically Weighted Approach. Sustainability,9(9), 1635. doi:10.3390/su9091635
* [Sperling, G. (n.d.). How Airbnb Combats Middle Class Stagnation(Tech.)](http://www.cedarcityutah.com/wp-content/uploads/2015/07/MiddleClassReport-MT-061915_r1.pdf)
* [Bion R, Chang R, Goodman J. (2017) How R helps Airbnb make the most of its data. PeerJ Preprints 5:e3182v1](https://doi.org/10.7287/peerj.preprints.3182v1)
* [Fu, J., Sun, J., & Wang, K. (2016). SPARK—A Big Data Processing Platform for Machine Learning.](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7823490)
