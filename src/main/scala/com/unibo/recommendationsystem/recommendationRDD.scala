package com.unibo.recommendationsystem

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession.builder

object recommendationRDD {
  def main(args: Array[String]): Unit = {

    //Initialize SparkSession
    val spark = builder
      .appName("recommendationsystem")
      .config("spark.master", "local[*]")
      .getOrCreate()

    val dataPathRec = "C:\\Users\\samue\\recommendationsystem\\steam-dataset\\recommendations.csv"
    val dataPathGames = "C:\\Users\\samue\\recommendationsystem\\steam-dataset\\games.csv"

    //PREPROCESSING
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

    // TF-IDF function definition
    def calculateTFIDF(userWordsDataset: RDD[(String, String)]): RDD[(String, Map[String, Double])] = {
      // Function to calculate term frequency (TF)
      def calculateTF(userWords: String): Map[String, Double] = {
        val wordsSplit = userWords.split(",")
        val totalWords = wordsSplit.size.toDouble
        val finalTF = wordsSplit.groupBy(identity).mapValues(_.length / totalWords)
        finalTF
      }

      // Function to calculate inverse document frequency (IDF)
      def calculateIDF(userWords: RDD[(String, String)]): Map[String, Double] = {
        val userCount = userWords.count()
        val wordsCount = userWords
          .flatMap { case (user, words) =>
            words.split(",").distinct.map(word => (user, word)) // Include docId and make words distinct
          }
          .map { case (user, word) => (word, user) } // Swap for grouping by word
          .groupByKey()
          .mapValues(_.toSet.size)

        val idfValues = wordsCount.map { case (word, count) => (word, math.log(userCount.toDouble / count)) }.collect().toMap[String,Double]
        idfValues
      }

      // Concatenates the words associated with each user into a comma-separated string
      val groupedUserWords = userWordsDataset
        .groupByKey() // Group by the userId
        .map { case (userId, wordsIterable) =>
          val wordsString = wordsIterable.mkString(",")
          (userId, wordsString)
        }

      val idfValues = calculateIDF(groupedUserWords)

      val tfidfValues = groupedUserWords.map { case (user, words) =>
        val termFrequencies = calculateTF(words)
        val tfidf = termFrequencies.map { case (word, tf) => (word, tf * idfValues.getOrElse(word, 0.0)) }
        (user, tfidf)
      }
      tfidfValues
    }

    val tfidfValues = calculateTFIDF(explodedRDD)

    val tTFIDFF = System.nanoTime()

    val tCosineSimilarityI = System.nanoTime()

    // Input: two vectors as a map of words and weights
    // Output: cosine similarity
    def computeCosineSimilarity(vector1: Map[String, Double], vector2: Map[String, Double]): Double = {
      def dotProduct(v1: Map[String, Double], v2: Map[String, Double]): Double = {
        v1.foldLeft(0.0) { case (acc, (key, value)) =>
          v2.get(key).map(value * _).getOrElse(0.0) + acc // Handle potential missing keys and type errors
        }
      }

      // Calculate vector magnitude (length)
      def magnitude(vector: Map[String, Double]): Double = {
        math.sqrt(vector.values.map(value => value * value).sum)
      }

      // Calculate cosine similarity
      dotProduct(vector1, vector2) / (magnitude(vector1) * magnitude(vector2))
    }

    // Get users similar to the target
    def getSimilarUsers(userId: Int, tfidfValues: RDD[(String, Map[String, Double])]): Array[(String, Double)] = {
      val userGames = tfidfValues.lookup(userId.toString).head
      tfidfValues.filter(_._1 != userId.toString).map {
        case (otherUserId, otherUserGames) => (otherUserId, computeCosineSimilarity(userGames, otherUserGames))
      }.collect().sortBy(-_._2).take(10)
    }

    val targetUser = 2591067
    val recommendations = getSimilarUsers(targetUser, tfidfValues)

    recommendations.foreach(println)

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

    finalRecommendations.take(30).foreach(println)

    println(s"\n\nExecution time(preprocessing): ${(tPreProcessingF - tPreProcessingI) / 1000000}ms\n")
    println(s"\n\nExecution time(Tf-Idf calculation): ${(tTFIDFF - tTFIDFI) / 1000000}ms\n")
    println(s"\n\nExecution time(Cosine similarity calculation): ${(tCosineSimilarityF - tCosineSimilarityI) / 1000000}ms\n")
    println(s"\n\nExecution time(final recommendation): ${(tFinalRecommendF - tFinalRecommendI) / 1000000}ms\n")
    println(s"\n\nExecution time(total): ${(tFinalRecommendF - tPreProcessingI) / 1000000}ms\n")
    spark.stop()
  }
}
