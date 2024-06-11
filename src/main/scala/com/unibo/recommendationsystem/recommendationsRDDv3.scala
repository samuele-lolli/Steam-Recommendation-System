package com.unibo.recommendationsystem

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.broadcast.Broadcast

object recommendationsRDDv3 {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .appName("RecommendationSystem")
      .config("spark.master", "local[*]")
      .getOrCreate()

    val dataPathRec = "C:\\Users\\samue\\recommendationsystem\\steam-dataset\\recommendations.csv"
    val dataPathGames = "C:\\Users\\samue\\recommendationsystem\\steam-dataset\\games.csv"

    val tPreProcessingI = System.nanoTime()

    val dfRec = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathRec)
    val dfGames = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathGames)

    // Select useful data from recommendations dataset
    val selectedRecRDD = dfRec.rdd.map(row => (row.getInt(0), row.getBoolean(4), row.getInt(6).toString))

    // Create a map of appId to game titles and broadcast it
    val titleDict = dfGames.rdd.map(row => (row.getInt(0), row.getString(1).toLowerCase.trim.replaceAll("\\s+", " "))).collectAsMap().toMap
    val broadcastTitleDict: Broadcast[Map[Int, String]] = spark.sparkContext.broadcast(titleDict)

    // Merge RDDs to include game titles
    val mergedRDD = selectedRecRDD.map {
      case (appId, _, user) =>
        (user, broadcastTitleDict.value.getOrElse(appId, "").split("\\s+"))
    }.groupByKey().mapValues(_.flatten.toArray).filter {
      case (_, words) => words.length >= 20
    }.flatMap {
      case (userId, words) => words.map(word => (userId, word))
    }

    val tPreProcessingF = System.nanoTime()

    val tTFIDFI = System.nanoTime()

    // Calculate TF-IDF
    def calculateTFIDF(userWordsDataset: RDD[(String, String)]): RDD[(String, Map[String, Double])] = {
      val userCount = userWordsDataset.map(_._1).distinct().count()

      // Calculate IDF values
      val idfValues = userWordsDataset.map { case (_, word) => (word, 1) }
        .reduceByKey(_ + _)
        .mapValues(count => math.log(userCount.toDouble / count))
        .collectAsMap().toMap

      val broadcastIdfValues: Broadcast[Map[String, Double]] = spark.sparkContext.broadcast(idfValues)

      // Calculate TF-IDF values
      userWordsDataset.groupByKey().mapValues { wordsIterable =>
        val words = wordsIterable.toSeq
        val totalWords = words.size.toDouble
        val termFrequencies = words.groupBy(identity).mapValues(_.size / totalWords)
        termFrequencies.map { case (word, tf) => (word, tf * broadcastIdfValues.value.getOrElse(word, 0.0)) }
      }
    }

    val tfidfValues = calculateTFIDF(mergedRDD)

    val tTFIDFF = System.nanoTime()

    val tCosineSimilarityI = System.nanoTime()

    // Compute Cosine Similarity
    def computeCosineSimilarity(vector1: Map[String, Double], vector2: Map[String, Double]): Double = {
      val dotProduct = vector1.keys.map(k => vector1(k) * vector2.getOrElse(k, 0.0)).sum
      val magnitude1 = math.sqrt(vector1.values.map(v => v * v).sum)
      val magnitude2 = math.sqrt(vector2.values.map(v => v * v).sum)
      dotProduct / (magnitude1 * magnitude2)
    }

    // Find similar users
    def getSimilarUsers(userId: Int, tfidfValues: RDD[(String, Map[String, Double])]): Array[(String, Double)] = {
      val userGames = tfidfValues.filter(_._1 == userId.toString).first()._2
      tfidfValues.filter(_._1 != userId.toString).map {
        case (otherUserId, otherUserGames) => (otherUserId, computeCosineSimilarity(userGames, otherUserGames))
      }.collect().sortBy(-_._2).take(10)
    }

    val targetUser = 2591067
    val recommendations = getSimilarUsers(targetUser, tfidfValues)

    val tCosineSimilarityF = System.nanoTime()

    // Get final recommendations
    val tFinalRecommendI = System.nanoTime()

    val titlesPlayedByTargetUser = mergedRDD.filter {
      case (user, _) => user == targetUser.toString
    }.map(_._2).distinct().collect().toSet

    val userIdsToFind = recommendations.take(3).map(_._1).toSet

    val finalRecommendations = selectedRecRDD.filter {
      case (_, isRecommended, user) =>
        userIdsToFind.contains(user) && isRecommended
    }.map {
      case (appId, _, user) => (appId, user, broadcastTitleDict.value.getOrElse(appId, ""))
    }.filter {
      case (_, _, title) => !titlesPlayedByTargetUser.contains(title)
    }.distinct()

    val tFinalRecommendF = System.nanoTime()

    finalRecommendations.take(20).foreach(println)

    println(s"\n\nExecution time(preprocessing): ${(tPreProcessingF - tPreProcessingI) / 1000000}ms\n")
    println(s"\n\nExecution time(Tf-Idf calculation): ${(tTFIDFF - tTFIDFI) / 1000000}ms\n")
    println(s"\n\nExecution time(Cosine similarity calculation): ${(tCosineSimilarityF - tCosineSimilarityI) / 1000000}ms\n")
    println(s"\n\nExecution time(final recommendation): ${(tFinalRecommendF - tFinalRecommendI) / 1000000}ms\n")
    println(s"\n\nExecution time(total): ${(tFinalRecommendF - tPreProcessingI) / 1000000}ms\n")

    spark.stop()
  }
}
