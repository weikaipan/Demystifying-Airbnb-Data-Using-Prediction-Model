#    Demystifying Airbnb Data Using Prediction Model

The analysis will identify opportunities and areas where properties can generate high gross margins by comparing the buying cost (annual mortgage) 
and expected annual revenue generated from both short term rentals on Airbnb and long-term rental price. The focus will be on visualizing up-and-
coming markets where the margins are high, as well as a comparison between long-term lease and short-term rentals. Along with these data, current 
regulations on Airbnb rentals in each city will also be taken into consideration. The project will use Hadoop and Pig to calculate aggregate statistics and 
discover which areas is recommended for long-term rental or short-term rental.

## Authors

* **Yi-Hsuan Fu**
* **Chun Hao Yang**
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

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.


See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the NYU License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Suzanne McIntosh
* [Inside Airbnb](http://insideairbnb.com/index.html)
* [NYPD Crime Statistics](http://www1.nyc.gov/site/nypd/stats/crime-statistics/crime-statistics-landing.page)
