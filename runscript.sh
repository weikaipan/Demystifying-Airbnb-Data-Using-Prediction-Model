# spark-submit src/dataprepare.py
cd ./src
python3 aggregatedata.py
cd ../spark/
PYTHONSTARTUP=./task.py pyspark
cd ../es/
python3 esearch.py
