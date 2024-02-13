package com.unibo.recommendationsystem

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.feature.{StringIndexer, IndexToString}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("recommendationsystem")
      .config("spark.master", "local")
      .getOrCreate()


    val csvFilePath = "C:\\Users\\User\\IdeaProjects\\recommendationsystem\\steam-dataset\\games.csv"

    // Leggi il file CSV come DataFrame
    val csvDataFrame: DataFrame = spark.read
      .format("csv")
      .option("header", "true") // Se la prima riga contiene i nomi delle colonne
      .option("inferSchema", "true") // Infere automaticamente i tipi di dati delle colonne
      .load(csvFilePath)

    // Mostra il DataFrame
    csvDataFrame.show()
    spark.stop()
  }
}