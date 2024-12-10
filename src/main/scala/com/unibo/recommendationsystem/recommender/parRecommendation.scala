package com.unibo.recommendationsystem.recommender

import com.unibo.recommendationsystem.utils.timeUtils
import scala.collection.parallel.ParMap

class parRecommendation(dataRec: Map[Int, Array[Int]], dataGames: Map[Int, String], dataMetaGames: Map[Int, Array[String]]) {

  /**
   * This method is the entry point for generating game recommendations for a given target user.
   * It orchestrates the entire process: preprocessing data, calculating TF-IDF, computing cosine similarities,
   * and generating final recommendations.
   *
   * @param targetUser The ID of the target user for whom the recommendations will be generated.
   */
  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (userTagsMap, userAppDetails) = timeUtils.time(preprocessData(), "Preprocessing Data", "PAR")

    println("Calculating term frequency and inverse document frequency...")
    val tfidfUserTags = timeUtils.time(calculateTFIDF(userTagsMap), "Calculating TF-IDF", "PAR")

    println("Calculating cosine similarity to find similar users...")
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(tfidfUserTags, targetUser), "Getting Similar Users", "PAR")

    println("Generating final recommendations...")
    timeUtils.time(generateFinalRecommendations(topUsersSimilarity, targetUser, userAppDetails), "Generating Recommendations", "PAR")
  }

  /**
   * Preprocesses the data by associating each user with their corresponding games, titles, and tags.
   * This step also prepares a mapping of user tags that will be used for calculating TF-IDF.
   *
   * @return A tuple consisting of:
   *         - A map of users and their concatenated tags.
   *         - A list of detailed user game information.
   */
  private def preprocessData(): (ParMap[Int, String], List[(Int, Int, String, Array[String])]) = {
    val userAppDetails = dataRec.par.flatMap { case (userId, appIds) =>
      appIds.flatMap { appId =>
        for {
          appTitle <- dataGames.get(appId)
          appTags <- dataMetaGames.get(appId)
        } yield (userId, appId, appTitle, appTags.map(_.trim.toLowerCase.replaceAll("\\s+", " ")))
      }
    }.filter(_._4.nonEmpty).toList

    val userTagsMap = userAppDetails
      .par
      .groupBy(_._1)
      .map { case (userId, details) =>
        userId -> details.flatMap(_._4).mkString(",")
      }

    (userTagsMap, userAppDetails)
  }

  /**
   * Calculates the Term Frequency-Inverse Document Frequency (TF-IDF) for each user based on the tags they have associated
   * with the games they played.
   *
   * @param userTagsMap A map of users and their corresponding tags.
   * @return A map of users and their TF-IDF values for each tag.
   */
  private def calculateTFIDF(userTagsMap: ParMap[Int, String]): ParMap[Int, Map[String, Double]] = {
    import scala.collection.compat._

    val totalUsers = userTagsMap.size

    // Calculate the inverse document frequency (IDF) for each tag
    val idfValues = userTagsMap.values
      .par
      .flatMap(_.split(",").distinct)
      .groupBy(identity)
      .seq.view
      .mapValues(tags => math.log(totalUsers.toDouble / tags.size))
      .toMap

    // For each user, calculate the term frequency (TF) and multiply it by the IDF for each tag
    userTagsMap.par.map { case (userId, tags) =>
      val tagList = tags.split(",")
      val tfValues = tagList.groupBy(identity).view.mapValues(_.length.toDouble / tagList.length).toMap
      userId -> tfValues.map { case (tag, tf) => tag -> tf * idfValues.getOrElse(tag, 0.0) }
    }
  }

  /**
   * Computes the cosine similarity between the target user's TF-IDF vector and the TF-IDF vectors of other users.
   *
   * @param tfidfUserTags A map of users and their corresponding TF-IDF vectors.
   * @param targetUser The ID of the target user for whom similarities will be calculated.
   * @return A list of the top 3 most similar users to the target user, sorted by similarity.
   */
  private def computeCosineSimilarity(tfidfUserTags: ParMap[Int, Map[String, Double]], targetUser: Int): List[Int] = {
    val targetVector = tfidfUserTags.getOrElse(targetUser, Map.empty)
    val targetMagnitude = math.sqrt(targetVector.values.map(v => v * v).sum)

    // Cosine similarity calculation for two vectors
    def cosineSimilarity(otherVector: Map[String, Double]): Double = {
      val dotProduct = targetVector.keySet.intersect(otherVector.keySet).view
        .map(tag => targetVector(tag) * otherVector(tag))
        .sum
      val otherMagnitude = math.sqrt(otherVector.values.map(v => v * v).sum)
      if (otherMagnitude == 0.0) 0.0 else dotProduct / (targetMagnitude * otherMagnitude)
    }

    tfidfUserTags
      .par
      .filterKeys(_ != targetUser)
      .map { case (userId, vector) => userId -> cosineSimilarity(vector) }
      .filter(_._2 > 0) // Filter out users with zero similarity
      .toList
      .sortBy(-_._2)
      .take(3)
      .map(_._1)
  }

  /**
   * Generates the final game recommendations for the target user based on the top similar users' preferences.
   * It filters out games that the target user has already played and recommends new ones.
   *
   * @param topUsers A list of the top similar users to the target user.
   * @param targetUser The ID of the target user.
   * @param userAppDetails A list of all user game details.
   */
  private def generateFinalRecommendations(
                                            topUsers: List[Int],
                                            targetUser: Int,
                                            userAppDetails: List[(Int, Int, String, Array[String])]
                                          ): Unit = {
    val targetUserGames = userAppDetails.collect {
      case (userId, gameId, _, _) if userId == targetUser => gameId
    }.toSet

    val recommendedGames = userAppDetails.par.collect {
      case (userId, gameId, gameTitle, _) if topUsers.contains(userId) && !targetUserGames.contains(gameId) =>
        (gameId, gameTitle, userId)
    }.groupBy(_._1)

    recommendedGames.foreach { case (gameId, gameDetails) =>
      val users = gameDetails.map(_._3).mkString(", ")
      println(s"Game ID: $gameId, Title: ${gameDetails.head._2}, Users: $users")
    }
  }
}
