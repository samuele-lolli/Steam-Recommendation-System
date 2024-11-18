package com.unibo.recommendationsystem.recommender

import com.unibo.recommendationsystem.utils.timeUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

class rddRecommendation(spark: SparkSession, dataRec: Dataset[Row], dataGames: DataFrame, metadata: DataFrame) {

  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (mergedRdd, explodedRDD, gamesData) = timeUtils.time(preprocessData(), "Preprocessing Data", "RDD")
    println("Calculate term frequency and inverse document frequency...")
    val tfidfValues = timeUtils.time(calculateTFIDF(explodedRDD), "Calculating TF-IDF", "RDD")
    println("Calculate cosine similarity to get similar users...")
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(targetUser, tfidfValues), "Getting Similar Users", "RDD")
    println("Calculate final recommendation...")
    timeUtils.time(generateFinalRecommendations(mergedRdd, topUsersSimilarity, gamesData, targetUser), "Generating Recommendations", "RDD")
  }

  private def preprocessData(): (RDD[(Int, String, String)], RDD[(String, String)], RDD[(Int, String)]) = {
    val selectedRecRDD = dataRec.rdd.map(row => (row.getInt(0), row.getInt(6).toString))

    val tags = metadata.rdd
      .map(row => (row.getInt(0), row.getList(2).toArray.map(_.toString).mkString(",").toLowerCase.replaceAll("\\s+", " ")))
      .collect()
      .toMap
/*
    val tags = metadata.rdd
      .map(row => (row.getInt(0), row.getList(2).toArray.map(_.toString).mkString(",").toLowerCase.replaceAll("\\s+", " ")))
      .collect()
      .toMap

 */
    val broadcastTagMap = spark.sparkContext.broadcast(tags)

    val mergedRDD = selectedRecRDD.map { case (appId, userId) =>
      val tag = broadcastTagMap.value.getOrElse(appId, "")
      (appId, tag, userId)
    }.filter(_._2.nonEmpty) //
      .cache()

    val aggregateDataRDD = mergedRDD
      .map { case (_, tag, user) => (user, tag.split(",")) }
      .reduceByKey(_ ++ _)
      .filter(_._2.length > 0)

    val explodedRDD = aggregateDataRDD.flatMap { case (userId, words) => words.map(word => (userId, word)) }

    val gamesTitlesRDD = dataGames.rdd.map(row => (row.getAs[Int]("app_id"), row.getAs[String]("title")))

    (mergedRDD, explodedRDD, gamesTitlesRDD)
  }

  private def calculateTFIDF(explodedRdd: RDD[(String, String)]): RDD[(String, Map[String, Double])] = {
    val calculateTF = (userWords: String) => {
      val wordsSplit = userWords.split(",")
      wordsSplit.groupBy(identity).mapValues(_.length.toDouble / wordsSplit.size)
    }

    val calculateIDF = (userWords: RDD[(String, String)]) => {
      val userCount = userWords.count()
      userWords.flatMap { case (_, words) => words.split(",").distinct }
        .map((_, 1))
        .reduceByKey(_ + _)
        .map { case (word, count) => (word, math.log(userCount.toDouble / count)) }
        .collect()
        .toMap
    }

    val groupedUserWords = explodedRdd
      .reduceByKey(_ + "," + _)
      .cache()

    val idfValues = calculateIDF(groupedUserWords)

    groupedUserWords.map { case (user, words) =>
      val tfValues = calculateTF(words)
      (user, tfValues.map { case (word, tf) => (word, tf * idfValues.getOrElse(word, 0.0)) })
    }
  }

  private def computeCosineSimilarity(
                               targetUser: Int,
                               tfidfValues: RDD[(String, Map[String, Double])]
                             ): Array[(String, Double)] = {

    val dotProduct = (v1: Map[String, Double], v2: Map[String, Double]) =>
      v1.foldLeft(0.0) { case (acc, (k, v)) => acc + v * v2.getOrElse(k, 0.0) }

    val magnitude = (vector: Map[String, Double]) => math.sqrt(vector.values.map(x => x * x).sum)

    val cosineSimilarity = (v1: Map[String, Double], v2: Map[String, Double]) =>
      dotProduct(v1, v2) / (magnitude(v1) * magnitude(v2))

    val targetUserGames = tfidfValues.filter(_._1 == targetUser.toString).map(_._2).collect().headOption

    targetUserGames.map { targetVector =>
      tfidfValues
        .filter(_._1 != targetUser.toString)
        .map { case (userId, vector) => (userId, cosineSimilarity(targetVector, vector)) }
        .collect()
        .sortBy(-_._2)
        .take(3)
    }.getOrElse(Array.empty)
  }

  private def generateFinalRecommendations(
                                 mergedRdd: RDD[(Int, String, String)],
                                 topUsersXSimilarity: Array[(String, Double)],
                                 gamesData: RDD[(Int, String)],
                                 targetUser: Int
                               ): Unit = {

    val appIdsPlayedByTargetUser = mergedRdd
      .filter(_._3 == targetUser.toString)
      .map(_._1)
      .collect()
      .toSet

    val userIdsToFind = topUsersXSimilarity.map(_._1).toSet

    val finalRecommendations = mergedRdd
      .filter { case (appId, _, user) => userIdsToFind.contains(user) && !appIdsPlayedByTargetUser.contains(appId) }
      .map { case (appId, _, user) => (appId, user) }
      .distinct()

    val recommendationsWithTitle = finalRecommendations
      .join(gamesData)
      .map { case (appId, (userId, title)) => (appId, title, userId) }
      .distinct()

    recommendationsWithTitle.collect().foreach { case (_, title, userId) =>
      println(s"userId: $userId, title: $title")
    }
  }
}
