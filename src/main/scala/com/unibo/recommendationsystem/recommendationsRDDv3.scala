package com.unibo.recommendationsystem

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

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

    // Create a map of appId to game titles
    val titleDict = dfGames.rdd.map(row => (row.getInt(0), row.getString(1).toLowerCase.trim.replaceAll("\\s+", " "))).collectAsMap()

    // Merge RDDs to include game titles
    val mergedRDD = selectedRecRDD.map {
      case (appId, recommended, user) =>
        (appId, recommended, user, titleDict.getOrElse(appId, ""))
    }

    // Aggregate data by user
    val aggregateDataRDD = mergedRDD.map {
      case (_, _, user, title) => (user, title.split("\\s+"))
    }.groupByKey().mapValues(_.flatten.toArray).filter {
      case (_, words) => words.length >= 20
    }

    // Explode the aggregated data for TF-IDF calculation
    val explodedRDD = aggregateDataRDD.flatMap {
      case (userId, words) => words.map(word => (userId, word))
    }

    val tPreProcessingF = System.nanoTime()

    val tTFIDFI = System.nanoTime()

    // Calculate TF-IDF
    def calculateTFIDF(userWordsDataset: RDD[(String, String)]): RDD[(String, Map[String, Double])] = {
      def calculateTF(words: Seq[String]): Map[String, Double] = {
        val totalWords = words.size.toDouble
        words.groupBy(identity).mapValues(_.size / totalWords)
      }

      def calculateIDF(userWords: RDD[(String, String)]): Map[String, Double] = {
        val userCount = userWords.count()
        userWords.map { case (_, word) => (word, 1) }.reduceByKey(_ + _).mapValues(count => math.log(userCount.toDouble / count)).collect().toMap
      }

      val idfValues = calculateIDF(userWordsDataset)
      userWordsDataset.groupByKey().mapValues { wordsIterable =>
        val termFrequencies = calculateTF(wordsIterable.toSeq)
        termFrequencies.map { case (word, tf) => (word, tf * idfValues.getOrElse(word, 0.0)) }
      }
    }

    val tfidfValues = calculateTFIDF(explodedRDD)

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
      val userGames = tfidfValues.lookup(userId.toString).head
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
      case (_, _, user, _) => user == targetUser.toString
    }.map(_._4).distinct().collect().toSet

    val userIdsToFind = recommendations.take(3).map(_._1).toSet

    val finalRecommendations = mergedRDD.filter {
      case (_, isRecommended, user, title) =>
        userIdsToFind.contains(user) && !titlesPlayedByTargetUser.contains(title) && isRecommended
    }.map {
      case (appId, _, user, title) => ((appId, title), user)
    }

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
