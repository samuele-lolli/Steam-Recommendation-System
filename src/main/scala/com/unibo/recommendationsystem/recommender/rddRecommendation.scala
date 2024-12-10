package com.unibo.recommendationsystem.recommender

import com.unibo.recommendationsystem.utils.timeUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

class rddRecommendation(spark: SparkSession, dataRec: Dataset[Row], dataGames: DataFrame, metadata: DataFrame) {

  /**
   * (RDD version) Generates personalized recommendations for a target user
   *
   * @param targetUser Int, The ID of the user for which we are generating recommendations
   */
  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (appIdUserDetails, userTagsGroup, gamesData) = timeUtils.time(preprocessData(), "Preprocessing Data", "RDD")

    println("Calculate term frequency and inverse document frequency...")
    val tfidfUserTags = timeUtils.time(calculateTFIDF(userTagsGroup), "Calculating TF-IDF", "RDD")

    println("Calculate cosine similarity to get similar users...")
    val top3SimilarUsers = timeUtils.time(computeCosineSimilarity(targetUser, tfidfUserTags), "Getting Similar Users", "RDD")

    println("Calculate final recommendation...")
    timeUtils.time(generateFinalRecommendations(appIdUserDetails, top3SimilarUsers, gamesData, targetUser), "Generating Recommendations", "RDD")
  }

  /**
   * Preprocesses the input data to create intermediate RDDs needed for further calculations
   *
   * @return A tuple of:
   *         - RDD[(Int, String, String)], Each app with its tags and associated user
   *         - RDD[(String, String)], Grouped tags for each user for tf-idf calculation
   *         - RDD[(Int, String)], Titles of the games
   */
  private def preprocessData(): (RDD[(Int, String, String)], RDD[(String, String)], RDD[(Int, String)]) = {

    val appIdUserId = dataRec.rdd.map(rec => (rec.getInt(0), rec.getInt(6).toString))

    //Extract tags for each appId
    val appIdTags = metadata.rdd
      .map(row => (row.getInt(0), row.getList(2).toArray.map(_.toString).mkString(",").toLowerCase.replaceAll("\\s+", " ")))
      .collect()
      .toMap

    val broadcastTagMap = spark.sparkContext.broadcast(appIdTags)

    //Add tags to games
    val appIdUserDetails = appIdUserId.map { case (appId, userId) =>
        val tags = broadcastTagMap.value.getOrElse(appId, "")
        (appId, tags, userId)
      }.filter(_._2.nonEmpty)
      .cache()

    //Get all tags together and remove all empty users
    val userTagsGroup = appIdUserDetails
      .flatMap { case (_, tags, userId) => tags.split(",").map(tag => (userId, tag)) }
      .reduceByKey(_ + "," + _)
      .filter(_._2.nonEmpty)
      .cache()

    //Extracts titles of the apps to use them in ifnal recommendation
    val gamesTitles = dataGames.rdd.map(row => (row.getAs[Int]("app_id"), row.getAs[String]("title")))

    (appIdUserDetails, userTagsGroup, gamesTitles)
  }

  /**
   * Computes TF-IDF values for all users based on their tags
   *
   * @param userTagsGroup RDD[(String, String)], userId with his grouped tags
   * @return RDD[(String, Map[String, Double])], tf-idf score map for each userId
   */
  private def calculateTFIDF(userTagsGroup: RDD[(String, String)]): RDD[(String, Map[String, Double])] = {
    def calculateTF(tags: String): Map[String, Double] = {
      val allTags = tags.split(",")
      allTags.groupBy(identity).mapValues(_.length.toDouble / allTags.size)
    }

    def calculateIDF(userTagsGroup: RDD[(String, String)]): Map[String, Double] = {
      val userCount = userTagsGroup.count()
      userTagsGroup.flatMap { case (_, tags) => tags.split(",").distinct }
        .map((_, 1))
        .reduceByKey(_ + _)
        .map { case (tag, count) => (tag, math.log(userCount.toDouble / count)) }
        .collect()
        .toMap
    }

    val idfValuesTag: Map[String, Double] = calculateIDF(userTagsGroup)

    val tfidfUserTags =  userTagsGroup.map { case (user, tags) =>
      val tfValues = calculateTF(tags)
      (user, tfValues.map { case (tag, tf) => (tag, tf * idfValuesTag.getOrElse(tag, 0.0)) })
    }

    tfidfUserTags
  }


  /**
   * Computes cosine similarity between the target user and all other users
   *
   * @param targetUser Int, the ID of the target user
   * @param tfidfUserTags RDD[(String, Map[String, Double])], tf-idf score map for each userId
   * @return Array[(String, Double)], the three userId with high cosine similarity score
   */
  private def computeCosineSimilarity(targetUser: Int, tfidfUserTags: RDD[(String, Map[String, Double])]): Array[(String, Double)] = {

    val tfIdfTargetUser = tfidfUserTags.filter(_._1 == targetUser.toString).map(_._2).collect().headOption

    //Computes the dot product of two vectors: multiplies the target user’s score for each tag by the other user’s score for the same
    def numerator = (targetScores: Map[String, Double], otherScores: Map[String, Double]) =>
      targetScores.foldLeft(0.0) { case (acc, (tag, score)) => acc + score * otherScores.getOrElse(tag, 0.0) }

    //Computes the magnitude of a vector: computes the square root of the sum of the squares of all tf-idf values for the vector
    def denominator = (scoresMap: Map[String, Double]) => math.sqrt(scoresMap.values.map(x => x * x).sum)

    def cosineSimilarity = (target: Map[String, Double], other: Map[String, Double]) =>
      numerator(target, other) / (denominator(target) * denominator(other))

    //Finds the top 3 similar users
    val top3SimilarUsers = tfIdfTargetUser.map { targetUserScores =>
      tfidfUserTags
        .filter(_._1 != targetUser.toString)
        .map { case (userId, otherUserScores) => (userId, cosineSimilarity(targetUserScores, otherUserScores)) }
        .collect()
        .sortBy(-_._2)
        .take(3)
    }.getOrElse(Array.empty)

    top3SimilarUsers
  }

  /**
   * Generates game recommendations for the target user based on similar users' preferences
   *
   * @param appIdUserDetails RDD[(Int, String, String)],  where each entry is (appId, tags, userId)
   * @param topUsersXSimilarity Array[(String, Double)], top similar users with similarity scores
   * @param gamesData RDD[(Int, String)], where each entry is (appId, game title)
   * @param targetUser Int, the ID of the target user
   */
  private def generateFinalRecommendations(appIdUserDetails: RDD[(Int, String, String)], topUsersXSimilarity: Array[(String, Double)], gamesData: RDD[(Int, String)], targetUser: Int): Unit = {
    val targetUserAppIds = appIdUserDetails
      .filter(_._3 == targetUser.toString)
      .map(_._1)
      .collect()
      .toSet

    val similarUsers = topUsersXSimilarity.map(_._1).toSet

    val finalRecommendations = appIdUserDetails
      .filter { case (appId, _, user) => similarUsers.contains(user) && !targetUserAppIds.contains(appId) }
      .map { case (appId, _, user) => (appId, user) }
      .distinct()

    val recommendationsWithTitle = finalRecommendations
      .join(gamesData)
      .map { case (appId, (userId, title)) => (appId, title, userId) }

    recommendationsWithTitle.collect().foreach { case (appId, title, userId) =>
      println(s"Game ID: $appId, userId: $userId, title: $title")
    }
  }
}
