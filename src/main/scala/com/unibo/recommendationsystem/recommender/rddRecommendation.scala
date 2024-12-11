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
    val userCount = userTagsGroup.count()
    val idfValuesTag = userTagsGroup
      .flatMap { case (_, tags) => tags.split(",").distinct }
      .map(tag => (tag, 1))
      .reduceByKey(_ + _)
      .map { case (tag, count) => (tag, math.log(userCount.toDouble / count)) } // Calcola IDF
      .collect()
      .toMap

    userTagsGroup.map { case (user, tags) =>
      val allTags = tags.split(",")
      val tfValues = allTags
        .groupBy(identity)
        .map { case (tag, occurrences) => (tag, occurrences.length.toDouble / allTags.length) } // Calcola TF

      val tfidfValues = tfValues.map { case (tag, tf) => (tag, tf * idfValuesTag.getOrElse(tag, 0.0)) }

      (user, tfidfValues)
    }
  }


  /**
   * Computes cosine similarity between the target user and all other users
   *
   * @param targetUser Int, the ID of the target user
   * @param tfidfUserTags RDD[(String, Map[String, Double])], tf-idf score map for each userId
   * @return Array[(String, Double)], the three userId with high cosine similarity score
   */
  private def computeCosineSimilarity(targetUser: Int, tfidfUserTags: RDD[(String, Map[String, Double])]): Array[(String, Double)] = {
    val targetUserScoresOpt = tfidfUserTags
      .filter(_._1 == targetUser.toString)
      .map(_._2)
      .collect()
      .headOption

    val targetUserScores = targetUserScoresOpt.get
    val broadcastTargetUserScores = spark.sparkContext.broadcast(targetUserScores)

    val similarities = tfidfUserTags
      .filter(_._1 != targetUser.toString)
      .mapPartitions(iter => {
        val localTargetScores = broadcastTargetUserScores.value
        iter.map { case (userId, otherUserScores) =>
          val numerator = localTargetScores.foldLeft(0.0) { case (acc, (tag, score)) =>
            acc + score * otherUserScores.getOrElse(tag, 0.0)
          }
          val targetMagnitude = math.sqrt(localTargetScores.values.map(x => x * x).sum)
          val otherMagnitude = math.sqrt(otherUserScores.values.map(x => x * x).sum)
          val denominator = targetMagnitude * otherMagnitude
          val similarity = if (denominator == 0.0) 0.0 else numerator / denominator
          (userId, similarity)
        }
      })
      .filter(_._2 > 0.0)

    similarities.top(3)(Ordering.by(_._2))
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
      .map { case (appId, (userId, title)) =>  (appId, (title, Set(userId))) }
      .reduceByKey { case ((title, userIds1), (_, userIds2)) =>
        (title, userIds1 ++ userIds2)
      }

    recommendationsWithTitle.foreach { case (appId, (title, userIds)) =>
      println(s"Game ID: $appId, userId: ${userIds.mkString(",")}, title: $title")
    }
  }
}


