# For Python3/Jupyter

## TUXML-basic.ipynb 

This script computes various statistics and executes some learning algorithms. 
For using it you need: 
TUXML_CSV_FILENAME="./config_bdd.csv"

As such a CSV file is rather big (2Go if we extract the whole database), we do not commit in the Git. 
Please contact us if you want some CSV files

## bdd2csv

This script extracts from the database a CSV file (columns are options and performance values such as size, time, etc.) 
Warning: it is a time-consuming procedure

## Prerequisistes 
Python 3
Jupyter http://jupyter.org/ (recommendations: nbconvert) 
pip3 install matplotlib pandas numpy scipy sklearn seaborn tpot 

# For R 

install.packages(pkg=c("ggplot2", "readr", "rpart", "rpart.plot", "randomForest", "caret", "gbm", "dplyr", "randomForestExplainer", "Metrics", "data.table"))  
