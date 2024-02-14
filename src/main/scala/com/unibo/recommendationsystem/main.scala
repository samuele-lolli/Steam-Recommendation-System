package com.unibo.recommendationsystem

import org.apache.spark
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("recommendationsystem")
      .config("spark.master", "local")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "512m")
      .getOrCreate()


    val csvFilePath = "C:\\Users\\User\\IdeaProjects\\recommendationsystem\\steam-dataset\\recommendations.csv"

    val rawData = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(csvFilePath)

    // Data Preprocessing
    val userIndexer = new StringIndexer()
      .setInputCol("user_id")
      .setOutputCol("user_id_indexed")
    val userIndexedData = userIndexer.fit(rawData).transform(rawData)

    val appIndexer = new StringIndexer()
      .setInputCol("app_id")
      .setOutputCol("app_id_indexed")
    val data = appIndexer.fit(userIndexedData).transform(userIndexedData)

//Train ALS Model
//    val als = new ALS()
//      .setUserCol("user_id_indexed")
//      .setItemCol("app_id_indexed")
//      .setRatingCol("hours")
//      .setRank(10)
//      .setMaxIter(10)
//      .setRegParam(0.1)
//      .setImplicitPrefs(true)
//
//    val model = als.fit(data)

    spark.stop()
  }
}