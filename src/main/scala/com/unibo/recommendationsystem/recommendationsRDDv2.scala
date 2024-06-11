package com.unibo.recommendationsystem

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.Map

object recommendationsRDDv2 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("recommendationsystem")
      .config("spark.master", "local[*]")
      .getOrCreate()

    // Percorsi dei dati
    val dataPathRec = "C:\\Users\\samue\\recommendationsystem\\steam-dataset\\recommendations.csv"
    val dataPathGames = "C:\\Users\\samue\\recommendationsystem\\steam-dataset\\games.csv"

    val tPreProcessingI = System.nanoTime()

    // Caricamento dei dataset
    val dfRec = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathRec)
    val dfGames = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathGames)

    val rddRec = dfRec.rdd
    val rddGames = dfGames.rdd

    // Selezione e trasformazione dei dati
    val selectedRecRDD = rddRec.map { row =>
      val appId = row.getInt(0)
      val recommended = row.getBoolean(4)
      val user = row.getInt(6).toString
      (appId, recommended, user)
    }

    val selectedGamesRDD = rddGames.map { row =>
      val appId = row.getInt(0)
      val title = row.getString(1)
      (appId, title)
    }

    // Broadcast del dizionario dei titoli
    val titleDict = spark.sparkContext.broadcast(selectedGamesRDD.collectAsMap())

    val mergedRDD = selectedRecRDD.map {
      case (appId, recommended, user) =>
        val title = titleDict.value.getOrElse(appId, "")
        (appId, recommended, user, title)
    }.cache()

    val cleanMergeRDD = mergedRDD.map { row =>
      val appId = row._1
      val recommended = row._2
      val user = row._3
      val title = row._4.toLowerCase().trim().replaceAll("\\s+", " ")
      (appId, recommended, user, title)
    }.cache()

    val t5 = System.nanoTime()


    val t2 = System.nanoTime()

    val datasetRDD = cleanMergeRDD.map { row =>
      val appId = row._1
      val recommended = row._2
      val user = row._3
      val words = row._4.split("\\s+")
      (appId, recommended, user, words)
    }.cache()

    val aggregateDataRDD = datasetRDD
      .map { case (_, _, user, words) => (user, words) }
      .groupByKey()
      .mapValues(_.flatten.toArray)
      .cache()

    val filteredDataRDD = aggregateDataRDD.filter { case (_, words) => words.length >= 20 }

    val explodedRDD = filteredDataRDD.flatMap { case (userId, words) =>
      words.map(word => (userId, word))
    }.cache()

    val tPreProcessingF = System.nanoTime()

    val tTFIDFI = System.nanoTime()

    def calculateTFIDF(userWordsDataset: RDD[(String, String)]): RDD[(String, Map[String, Double])] = {
      def calculateTF(words: Array[String]): Map[String, Double] = {
        val totalWords = words.length.toDouble
        words.groupBy(identity).mapValues(_.length / totalWords)
      }

      def calculateIDF(userWords: RDD[(String, Array[String])]): Map[String, Double] = {
        val numDocs = userWords.count()
        val docFrequencies = userWords.flatMap { case (_, words) =>
          words.distinct.map(word => (word, 1))
        }.reduceByKey(_ + _).collectAsMap()

        docFrequencies.map { case (word, count) => (word, math.log(numDocs.toDouble / count)) }
      }

      val userWordsGrouped = userWordsDataset.groupByKey().mapValues(_.toArray).cache()

      val idfValues = calculateIDF(userWordsGrouped)

      userWordsGrouped.map { case (user, words) =>
        val tfValues = calculateTF(words)
        val tfidfValues = tfValues.map { case (word, tf) => (word, tf * idfValues.getOrElse(word, 0.0)) }
        (user, tfidfValues)
      }
    }

    val tfidfValues = calculateTFIDF(explodedRDD).cache()

    val tTFIDFF = System.nanoTime()

    val tCosineSimilarityI = System.nanoTime()

    def computeCosineSimilarity(vector1: Map[String, Double], vector2: Map[String, Double]): Double = {
      def dotProduct(v1: Map[String, Double], v2: Map[String, Double]): Double = {
        v1.foldLeft(0.0) { case (acc, (key, value)) =>
          acc + v2.getOrElse(key, 0.0) * value
        }
      }

      def magnitude(vector: Map[String, Double]): Double = {
        math.sqrt(vector.values.map(x => x * x).sum)
      }

      dotProduct(vector1, vector2) / (magnitude(vector1) * magnitude(vector2))
    }

    val targetUser = 2591067
    val t0 = System.nanoTime()

    def getSimilarUsers(userId: Int, tfidfValues: RDD[(String, Map[String, Double])]): Array[(String, Double)] = {
      val userGames = tfidfValues.filter(_._1 == userId.toString).first()._2

      tfidfValues.filter(_._1 != userId.toString)
        .map { case (otherUserId, otherUserGames) =>
          (otherUserId, computeCosineSimilarity(userGames, otherUserGames))
        }
        .collect()
        .sortBy(-_._2)
        .take(10)
    }

    val recommendations = getSimilarUsers(targetUser, tfidfValues)
    val tCosineSimilarityF = System.nanoTime()
    recommendations.foreach(println)


    val tFinalRecommendI = System.nanoTime()

    val titlesPlayedByTargetUserRDD = cleanMergeRDD
      .filter { case (_, _, user, _) => user == targetUser.toString }
      .map { case (_, _, _, title) => title }
      .distinct()

    val titlesPlayedByTargetUserSet = titlesPlayedByTargetUserRDD.collect().toSet

    val userIdsToFind = recommendations.take(3).map(_._1).toSet

    val filteredRDD = cleanMergeRDD.filter { case (_, isRecommended, user, title) =>
      userIdsToFind.contains(user) && isRecommended && !titlesPlayedByTargetUserSet.contains(title)
    }

    val finalRecommendations = filteredRDD.map { case (appId, _, user, title) =>
      ((appId, title), user)
    }

    val tFinalRecommendF = System.nanoTime()

    finalRecommendations.take(20).foreach(println)

    // Calculating execution times
    println("\n\nExecution time(preprocessing):\t"+ (tPreProcessingF-tPreProcessingI)/1000000 + "ms\n")
    println("\n\nExecution time(Tf-Idf calculation):\t"+ (tTFIDFF-tTFIDFI)/1000000 + "ms\n")
    println("\n\nExecution time(Cosine similarity calculation):\t"+ (tCosineSimilarityF-tCosineSimilarityI)/1000000 + "ms\n")
    println("\n\nExecution time(final recommendation):\t"+ (tFinalRecommendF-tFinalRecommendI)/1000000 + "ms\n")
    println("\n\nExecution time(total):\t"+ (tFinalRecommendF-tPreProcessingI)/1000000 + "ms\n") //da sistemare

    spark.stop()
  }
}
