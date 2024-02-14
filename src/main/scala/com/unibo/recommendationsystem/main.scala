package com.unibo.recommendationsystem

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession

object main {
  def main(args: Array[String]): Unit = {

    /* Inizializzazione SparkSession */
    val spark = SparkSession
      .builder
      .appName("recommendationsystem")
      .config("spark.master", "local")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "512m")
      .getOrCreate()

    /* Path per il dataset */
    val csvFilePath = "C:\\Users\\User\\IdeaProjects\\recommendationsystem\\steam-dataset\\recommendations.csv"

    /* Caricamento dei dati nella forma originale */
    val rawData = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(csvFilePath)

    /* Pre-processing dei dati (a cosa serve sta roba? Mistero..) */
    val userIndexer = new StringIndexer()
      .setInputCol("user_id")
      .setOutputCol("user_id_indexed")
    val userIndexedData = userIndexer.fit(rawData).transform(rawData)

    val appIndexer = new StringIndexer()
      .setInputCol("app_id")
      .setOutputCol("app_id_indexed")
    val data = appIndexer.fit(userIndexedData).transform(userIndexedData)

    /* Inizializzazione e training del modello ALS (KNN) */
    println("INIZIO DEL TRAINING DI ALS MODEL.........")
    val als = new ALS()
      .setUserCol("user_id_indexed")
      .setItemCol("app_id_indexed")
      .setRatingCol("hours")
      .setRank(10)
      .setMaxIter(10)
      .setRegParam(0.1)
      .setImplicitPrefs(true)

    val model = als.fit(data)

    /* Generazione di 5 raccomandazioni per l'utente con id=0 */
    import spark.implicits._
    val userId = 0
    val userSubset = Seq(userId).toDF("user_id_indexed") // Create the DataFrame
    val recommendations = model.recommendForUserSubset(userSubset, 5)
    recommendations.show()
    spark.stop()
  }
}