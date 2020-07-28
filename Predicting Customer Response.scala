// Databricks notebook source
// MAGIC %md
// MAGIC 
// MAGIC # Predicting Customer Response to Bank Direct Telemarketing Campaign
// MAGIC 
// MAGIC Telemarketing advertising campaigns is a billion dollar effort and one of the central uses of machine learning model. However, its data and methods are usually kept under lock and key. The Project is related with direct marketing campaigns of a banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.
// MAGIC 
// MAGIC Direct marketing enables banks and other financial organizations  to  focus  on  those  customers  who  present  the likelihood  of  subscribing  to  their  products,  offers,  and  other packages.  Most  often  than  not,  identifying  these  group  of customers poses  a  challenge to financial  institutions.  In line with the  aforementioned,  this  study  considered  the  typical  case  of bank  direct  marketing  campaign  dataset  with main objectives.  
// MAGIC 
// MAGIC To predict customer response  to  bank  direct marketing  by  applying  two classifiers  namely Decision  Tree and Logistic  Regression Model. 

// COMMAND ----------

// MAGIC %md ### Load Source Data
// MAGIC The data for this Project is provided as a CSV file containing Customer details we need to predict if the customer will subscribe a term deposit . 
// MAGIC 
// MAGIC You will load this data into a DataFrame and display it.

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val bankDF = sqlContext.read.format("csv")
// MAGIC   .option("header", "true")
// MAGIC   .option("inferSchema", "true")
// MAGIC   .option("delimiter", ";")
// MAGIC   .load("/FileStore/tables/original.csv")
// MAGIC 
// MAGIC display(bankDF)

// COMMAND ----------

// DBTITLE 1,Print Schema
// MAGIC %scala
// MAGIC 
// MAGIC bankDF.printSchema();

// COMMAND ----------

// DBTITLE 1,Creating Temp View from Dataframe 
// MAGIC %scala
// MAGIC 
// MAGIC bankDF.createOrReplaceTempView("BankData")

// COMMAND ----------

// DBTITLE 1,Querying the Temporary View
// MAGIC %sql
// MAGIC 
// MAGIC select * from BankData;

// COMMAND ----------

// MAGIC %md
// MAGIC # Data Details:
// MAGIC 
// MAGIC Input variables:
// MAGIC 
// MAGIC ### Bank client data:
// MAGIC 1 - age (numeric)
// MAGIC 
// MAGIC 2 - job : type of job (categorical: 'admin.','blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'selfemployed', 'services', 'student', 'technician', 'unemployed','unknown')
// MAGIC 
// MAGIC 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
// MAGIC 
// MAGIC 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
// MAGIC 
// MAGIC 5 - default: has credit in default? (categorical: 'no','yes','unknown')
// MAGIC 
// MAGIC 6 - housing: has housing loan? (categorical: 'no','yes','unknown')
// MAGIC 
// MAGIC 7 - loan: has personal loan? (categorical: 'no','yes','unknown')
// MAGIC 
// MAGIC ### Related with the last contact of the current campaign:
// MAGIC 8 - contact: contact communication type (categorical: 'cellular','telephone')
// MAGIC 
// MAGIC 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
// MAGIC 
// MAGIC 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
// MAGIC 
// MAGIC 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
// MAGIC 
// MAGIC ### Other attributes:
// MAGIC 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
// MAGIC 
// MAGIC 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
// MAGIC 
// MAGIC 14 - previous: number of contacts performed before this campaign and for this client (numeric)
// MAGIC 
// MAGIC 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
// MAGIC 
// MAGIC ### social and economic context attributes
// MAGIC 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
// MAGIC 
// MAGIC 17 - cons.price.idx: consumer price index - monthly indicator (numeric)
// MAGIC 
// MAGIC 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
// MAGIC 
// MAGIC 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
// MAGIC 
// MAGIC 20 - nr.employed: number of employees - quarterly indicator (numeric)
// MAGIC 
// MAGIC ### Output variable (desired target):
// MAGIC 
// MAGIC 21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC #Exploratory Data Analysis

// COMMAND ----------

// DBTITLE 1,Distribution of our Labels:
// MAGIC %md
// MAGIC 
// MAGIC This is an important aspect that will be further discussed is dealing with imbalanced dataset. Knowing that we are dealing with an imbalanced dataset will help us determine what will be the best approach to implement our predictive model.

// COMMAND ----------

// DBTITLE 1,Displaying the Number(count) of Subscribers to Product (Yes/No)
// MAGIC %sql
// MAGIC 
// MAGIC select y as Subscribe_To_Product, count(y) as Subscribe_To_Product_Count from BankData group by y;

// COMMAND ----------

// DBTITLE 1,Displaying Percentage Number(count) of Subscribers to Product 
// MAGIC %sql
// MAGIC 
// MAGIC select y as Subscribe_To_Product, count(y) as Subscribe_To_Product_Count from BankData group by y;

// COMMAND ----------

// DBTITLE 1,Gender Analysis
// MAGIC %md
// MAGIC 
// MAGIC #### Questions to ask Ourselves:
// MAGIC 
// MAGIC * What is the age distribution between data? Are there any significant discrepancies?.
// MAGIC 
// MAGIC #### Summary:
// MAGIC 
// MAGIC * Age by Gender: The average age of customer is 40.02

// COMMAND ----------

// DBTITLE 1,Age 
// MAGIC %sql
// MAGIC 
// MAGIC select age, count(age) from BankData group by age order by age;

// COMMAND ----------

// DBTITLE 1,Mean Age
// MAGIC %sql
// MAGIC 
// MAGIC select mean(age) from BankData;

// COMMAND ----------

// DBTITLE 1,Jobs
// MAGIC %sql
// MAGIC 
// MAGIC select job, count(job) from BankData group by job;

// COMMAND ----------

// DBTITLE 1,Marital Status
// MAGIC %sql
// MAGIC 
// MAGIC select marital, count(marital) from BankData group by marital;

// COMMAND ----------

// DBTITLE 1,Education Background
// MAGIC %sql
// MAGIC 
// MAGIC select education, count(education) from BankData group by education;

// COMMAND ----------

// DBTITLE 1,Having House Loan?
// MAGIC %sql 
// MAGIC 
// MAGIC select housing, count(housing) from BankData group by housing;

// COMMAND ----------

// DBTITLE 1,Having Prersonal Loan?
// MAGIC %sql
// MAGIC 
// MAGIC select loan, count(loan) from BankData group by loan;

// COMMAND ----------

// MAGIC %md ## Creating a Regression Model
// MAGIC 
// MAGIC In this Project, you will implement a regression model that will predict the Customer Response based on many attributes available in Banking Telemarketing Data
// MAGIC 
// MAGIC ### Import Spark SQL and Spark ML Libraries
// MAGIC 
// MAGIC First, import the libraries you will need:

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC import org.apache.spark.sql.functions._
// MAGIC import org.apache.spark.sql.Row
// MAGIC import org.apache.spark.sql.types._
// MAGIC 
// MAGIC import org.apache.spark.ml.classification.LogisticRegression
// MAGIC import org.apache.spark.ml.feature.VectorAssembler

// COMMAND ----------

// MAGIC %md ### Prepare the Training Data
// MAGIC To train the regression model, you need a training data set that includes a vector of numeric features, and a label column. In this project, you will use the **VectorAssembler** class to transform the feature columns into a vector, and then rename the **Y** column to **label**.

// COMMAND ----------

// MAGIC %md ###VectorAssembler()
// MAGIC 
// MAGIC VectorAssembler():  is a transformer that combines a given list of columns into a single vector column. It is useful for combining raw features and features generated by different feature transformers into a single feature vector, in order to train ML models like logistic regression and decision trees. 
// MAGIC 
// MAGIC **VectorAssembler** accepts the following input column types: **all numeric types, boolean type, and vector type.** 
// MAGIC 
// MAGIC In each row, the **values of the input columns will be concatenated into a vector** in the specified order.

// COMMAND ----------

// DBTITLE 1,List all String Data Type Columns in an Array in further processing
// MAGIC %scala
// MAGIC 
// MAGIC var StringfeatureCol = Array("job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome", "y")

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ###StringIndexer
// MAGIC 
// MAGIC StringIndexer encodes a string column of labels to a column of label indices.

// COMMAND ----------

// DBTITLE 1,Example of StringIndexer
import org.apache.spark.ml.feature.StringIndexer

val df = spark.createDataFrame(
  Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
).toDF("id", "category")

val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")

val indexed = indexer.fit(df).transform(df)

display(indexed)

// COMMAND ----------

// MAGIC %md ### Define the Pipeline
// MAGIC A predictive model often requires multiple stages of feature preparation. 
// MAGIC 
// MAGIC A pipeline consists of a series of *transformer* and *estimator* stages that typically prepare a DataFrame for modeling and then train a predictive model. 
// MAGIC 
// MAGIC In this case, you will create a pipeline with stages:
// MAGIC 
// MAGIC - A **StringIndexer** estimator that converts string values to indexes for categorical features
// MAGIC - A **VectorAssembler** that combines categorical features into a single vector

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC import org.apache.spark.ml.attribute.Attribute
// MAGIC import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
// MAGIC import org.apache.spark.ml.{Pipeline, PipelineModel}
// MAGIC 
// MAGIC val indexers = StringfeatureCol.map { colName =>
// MAGIC   new StringIndexer().setInputCol(colName).setOutputCol(colName + "_indexed")
// MAGIC }
// MAGIC 
// MAGIC val pipeline = new Pipeline()
// MAGIC                     .setStages(indexers)      
// MAGIC 
// MAGIC val bankingDF = pipeline.fit(bankDF).transform(bankDF)

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC bankingDF.printSchema()

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC bankingDF.show()

// COMMAND ----------

// MAGIC %md ### Split the Data
// MAGIC It is common practice when building supervised machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this project, you will use 70% of the data for training, and reserve 30% for testing. In the testing data, the **label** column is renamed to **trueLabel** so you can use it later to compare predicted labels with known actual values.

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val splits = bankingDF.randomSplit(Array(0.7, 0.3))
// MAGIC val train = splits(0)
// MAGIC val test = splits(1)
// MAGIC val train_rows = train.count()
// MAGIC val test_rows = test.count()
// MAGIC println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

// COMMAND ----------

// DBTITLE 1,VectorAssembler() that combines categorical features into a single vector
// MAGIC %scala
// MAGIC 
// MAGIC val assembler = new VectorAssembler().setInputCols(Array("age", "duration", "campaign", "pdays", "previous", "empvarrate", "conspriceidx", "consconfidx", "euribor3m", 
// MAGIC "nremployed", "job_indexed", "marital_indexed", "education_indexed", "default_indexed", "housing_indexed", "loan_indexed", "contact_indexed", "month_indexed", "day_of_week_indexed", "poutcome_indexed")).setOutputCol("features")
// MAGIC 
// MAGIC val training = assembler.transform(train).select($"features", $"y_indexed".alias("label"))
// MAGIC 
// MAGIC training.show()

// COMMAND ----------

// MAGIC %md ### Train a Regression Model
// MAGIC Next, you need to train a regression model using the training data. To do this, create an instance of the regression algorithm you want to use and use its **fit** method to train a model based on the training DataFrame. In this Project, you will use a *Logistic Regression* algorithm - though you can use the same technique for any of the regression algorithms supported in the spark.ml API.

// COMMAND ----------

// MAGIC %scala
// MAGIC import org.apache.spark.ml.classification.LogisticRegression
// MAGIC 
// MAGIC val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setRegParam(0.3)
// MAGIC val model = lr.fit(training)
// MAGIC println("Model Trained!")

// COMMAND ----------

// MAGIC %md ### Prepare the Testing Data
// MAGIC Now that you have a trained model, you can test it using the testing data you reserved previously. First, you need to prepare the testing data in the same way as you did the training data by transforming the feature columns into a vector. This time you'll rename the **y_indexed** column to **trueLabel**.

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val testing = assembler.transform(test).select($"features", $"y_indexed".alias("trueLabel"))
// MAGIC testing.show()

// COMMAND ----------

// MAGIC %md ### Test the Model
// MAGIC Now you're ready to use the **transform** method of the model to generate some predictions. But in this case you are using the test data which includes a known true label value, so you can compare the predicted Revenue. 

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val prediction = model.transform(testing)
// MAGIC val predicted = prediction.select("features", "prediction", "trueLabel")
// MAGIC predicted.show()

// COMMAND ----------

// MAGIC %md ### Model Evalation
// MAGIC 
// MAGIC spark.mllib comes with a number of machine learning algorithms that can be used to learn from and make predictions on data. When these algorithms are applied to build machine learning models, there is a need to evaluate the performance of the model on some criteria, which depends on the application and its requirements. spark.mllib also provides a suite of metrics for the purpose of evaluating the performance of machine learning models.

// COMMAND ----------

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val evaluator = new BinaryClassificationEvaluator().setLabelCol("trueLabel").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
val auc = evaluator.evaluate(prediction)
println("AUC = " + (auc))

// COMMAND ----------

// MAGIC %md ### Train a Classification Model (Decision tree classifier)
// MAGIC Next, you need to train a Classification Model using the training data. To do this, create an instance of the Decision tree classifier algorithm you want to use and use its **fit** method to train a model based on the training DataFrame. In this Project, you will use a *Decision tree classifier* algorithm 

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC import org.apache.spark.ml.classification.DecisionTreeClassificationModel
// MAGIC import org.apache.spark.ml.classification.DecisionTreeClassifier
// MAGIC import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// MAGIC 
// MAGIC val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")
// MAGIC 
// MAGIC val model = dt.fit(training)
// MAGIC 
// MAGIC println("Model Trained!")

// COMMAND ----------

// MAGIC %md ### Test the Model
// MAGIC Now you're ready to use the **transform** method of the model to generate some predictions. But in this case you are using the test data which includes a known true label value, so you can compare the predicted Attrition. 

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val prediction = model.transform(testing)
// MAGIC val predicted = prediction.select("features", "prediction", "trueLabel")
// MAGIC predicted.show(100)

// COMMAND ----------

// MAGIC %md ### Classification model Evalation
// MAGIC 
// MAGIC spark.mllib comes with a number of machine learning algorithms that can be used to learn from and make predictions on data. When these algorithms are applied to build machine learning models, there is a need to evaluate the performance of the model on some criteria, which depends on the application and its requirements. spark.mllib also provides a suite of metrics for the purpose of evaluating the performance of machine learning models.

// COMMAND ----------

// MAGIC %scala
// MAGIC 
// MAGIC val evaluator = new MulticlassClassificationEvaluator()
// MAGIC   .setLabelCol("trueLabel")
// MAGIC   .setPredictionCol("prediction")
// MAGIC   .setMetricName("accuracy")
// MAGIC val accuracy = evaluator.evaluate(prediction)
