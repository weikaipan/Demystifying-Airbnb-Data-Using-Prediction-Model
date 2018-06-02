# spark-submit src/dataprepare.py
cd ./spark/
PYTHONSTARTUP=./task.py pyspark
cd ../es/
python3 esearch.py
