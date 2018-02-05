JAVA=javac
HADOOP=hdfs
HFLAG=dfs
CLASS=-classpath
YARN=`yarn classpath`
DFLAGS=-d
FILEPATH=/user/wkp219/class1

put:
	$(HADOOP) $(HFLAG) -put $(FILE) $(FILEPATH)/ 

check:
	$(HADOOP) $(HFLAG) -ls $(FILEPATH)

peek:
	$(HADOOP) $(HFLAG) -cat $(FILEPATH)/listings.csv

clean:
	$(HADOOP) $(HFLAG) -rm -r $(TARGET)

show:
	hdfs dfs -cat /user/wkp219/class1/output/*
