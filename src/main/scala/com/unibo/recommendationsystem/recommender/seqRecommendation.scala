package com.unibo.recommendationsystem.recommender

import com.unibo.recommendationsystem.utils.timeUtils
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods
import scala.collection.compat.{toMapViewExtensionMethods, toTraversableLikeExtensionMethods}
import scala.io.Source
import scala.util.Using

class seqRecommendation(dataRecPath: String, dataGamesPath: String, metadataPath: String) {

  private val dataRec: Map[Int, Array[Int]] =  loadRecommendations(dataRecPath)
  private val dataGames: Map[Int, String] = loadDataGames(dataGamesPath)
  private val metadata: Map[Int, Array[String]] = loadMetadata(metadataPath)

  /**
   * (Sequential version) Generates personalized recommendations for a target user
   *
   * @param targetUser Int, The ID of the user for which we are generating recommendations
   */
  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (userTagsMap, userAppDetails) = timeUtils.time(preprocessData(), "Preprocessing Data", "Seq")
    println("Calculate term frequency and inverse document frequency...")
    val tfidfUserTags = timeUtils.time(calculateTFIDF(userTagsMap), "Calculating TF-IDF", "Seq")
    println("Calculate cosine similarity to get similar users...")
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(tfidfUserTags, targetUser), "Getting Similar Users", "Seq")
    println("Calculate final recommendation...")
    timeUtils.time(generateFinalRecommendations(topUsersSimilarity, targetUser, userAppDetails), "Generating Recommendations", "Seq")
  }

  /**
   * Preprocesses the input data to create intermediate scala.collections needed for further calculations.
   *
   * @return A tuple of:
   *         - Map[Int, String], map each user with his tags for tf-idf calculation
   *         - List[(Int, Int, String, Array[String])], full user-item metadata
   */
  private def preprocessData(): (Map[Int, String], List[(Int, Int, String, Array[String])]) = {
    //Combines user, game, and metadata details, filtering out empty tags
    val userAppDetails = dataRec.flatMap {
      case (userId, appIds) =>
        for {
          appId <- appIds
          appTitle <- dataGames.get(appId)
          appTags <- metadata.get(appId)
        } yield (userId, appId, appTitle, appTags.map(_.trim.toLowerCase.replaceAll("\\s+", " ")))
    }.filter(_._4.nonEmpty)
    /*
     * (8897055,616560,Ultimate Epic Battle Simulator,[Ljava.lang.String;@60fe75f7)
     */

    // Maps userIds to their list of tags and filters out empty lists
    val userTags = userAppDetails.map {
      case (userId, _, _, appTags) =>
        (userId, appTags.mkString(",").split(",").filter(_.nonEmpty).toList)
    }.filter(_._2.nonEmpty)
    /*
     * (8897055,List(simulation, sandbox, war, strategy, action, funny, medieval, violent, singleplayer, gore, indie, early access, fantasy, open world, zombies, adventure, survival, memes, third person, atmospheric))
     */

    //Aggregates all tags for each user
    val groupedUserTags: Map[Int, List[String]] = userTags.groupMapReduce(userId => userId._1)(data => data._2)((tags1, tags2) => tags1 ++ tags2)
    /*
     * (8897055,List(simulation, sandbox, war, strategy, action, funny, medieval, violent, singleplayer, gore, indie, early access, fantasy, open world, zombies, adventure, survival, memes, third person, atmospheric, visual novel, dating sim, casual, sexual content, cute, 2d, anime, story rich, choices matter, romance, multiple endings, demons, mature, comedy, singleplayer, adventure, nudity, hentai, shooter, psychological horror, rpg, action, adventure, anime, sexual content, visual novel, female protagonist, nudity, cute, jrpg, singleplayer, naval, world war ii, third-person shooter, military, fps, shooter, hentai, naval combat, mature, sexual content, visual novel, casual, anime, adventure, simulation, nudity, memes, comedy, nsfw))
     */
    //Exploded tags into (userId, tag)
    val explodeUserTag: List[(Int, String)] = groupedUserTags.toList
      .flatMap { case (userId, tags) => tags.map(tag => (userId, tag)) }
    /*
     * (8897055,simulation)
     * (8897055,sandbox)
     * (8897055,war)
     * (8897055,strategy)
     * (8897055,action)
     */

    //For each user groups all his tags separated by commas and filters out users without tags
    val userTagsMap: Map[Int, String] = explodeUserTag
      .groupBy(_._1) // Group by user ID
      .mapValues(_.map(_._2).mkString(",")) // Concatenate all strings for each user


    (userTagsMap, userAppDetails.toList)
  }

  /**
   * Computes TF-IDF values for all users based on their tags
   *
   * @param userTagsMap Map[Int, String], map of userIDs to concatenated strings of their tags
   * @return Map[Int, Map[String, Double] ], A map where each user ID maps to another map of tags and their respective TF-IDF scores
   */
  private def calculateTFIDF(userTagsMap: Map[Int, String]): Map[Int, Map[String, Double]] = {
    //Takes user's tags as input and calculates the Term Frequency for each tag
    val calculateTF = (tags: String) => {
      val allTags = tags.split(",")
      allTags.groupBy(identity).mapValues(_.length.toDouble / allTags.length)
    }

    //Computes the Inverse Document Frequency for each tag
    val calculateIDF = (userTagsMap: Map[Int, String]) => {
      val userCount = userTagsMap.size
      userTagsMap.values
        .flatMap(_.split(",").distinct)
        .groupBy(identity)
        .map { case (tag, occurrences) => (tag, math.log(userCount.toDouble / occurrences.size)) }
    }

    val idfValuesTag: Map[String, Double] = calculateIDF(userTagsMap)

    //Calculates the TF-IDF values for each tag for every user
    userTagsMap
      .map { case (user, words) =>
        val tfValues = calculateTF(words)
        user -> tfValues.map { case (word, tf) => word -> tf * idfValuesTag.getOrElse(word, 0.0) }
      }

  }

  /**
   * Computes cosine similarity between the target user and all other users
   *
   * @param tfidfUserTags Map[Int, Map[String, Double] ], tf-idf score map for each userId
   * @param targetUser Int, the ID of the target user
   * @return A list of the top 3 most similar user IDs
   */
  private def computeCosineSimilarity(tfidfUserTags: Map[Int, Map[String, Double]], targetUser: Int): List[Int] = {

    def cosineSimilarity(targetUserScores: Map[String, Double], otherUserScores: Map[String, Double], tUserDenominator: Double): Double = {
      //Computes the dot product of two vectors
      val numerator = targetUserScores.keySet.intersect(otherUserScores.keySet).foldLeft(0.0) { (acc, k) => acc + targetUserScores(k) * otherUserScores(k) }
      //Computes the product of magnitudes of the two vectors)
      val denominator = tUserDenominator * math.sqrt(otherUserScores.values.map(v => v * v).sum)
      if (denominator == 0) 0.0 else numerator / denominator
    }

    val targetUserScores = tfidfUserTags(targetUser)
    //Calculates the magnitude of the target user's vector
    val tUserDenominator = math.sqrt(targetUserScores.values.map(v => v * v).sum)

    //Filter out users with zero TF-IDF vectors
    val nonZeroTfidf = tfidfUserTags.filter { case (_, vector) => math.sqrt(vector.values.map(v => v * v).sum) != 0 }

    //Find the top 3 similar users
    val topUsers = nonZeroTfidf.view
      .filterKeys(_ != targetUser)
      .mapValues(cosineSimilarity(targetUserScores, _, tUserDenominator))
      .toList.sortBy(-_._2)
      .take(3)
      .map(_._1)

    topUsers
  }

  /**
   * Generates and prints final game recommendations for a target user based on games played by similar users
   *
   * @param topUsers List[Int], list of IDs of the most similar users
   * @param targetUser Int, the ID of the target user
   * @param userAppDetails List[(Int, Int, String, Array[String])], list containing metadata about user-game recommendations
   */
  private def generateFinalRecommendations(topUsers: List[Int], targetUser: Int, userAppDetails: List[(Int, Int, String, Array[String])]): Unit = {
    //Get all games played by the target user
    val gamesByTargetUser = userAppDetails.filter(_._1 == targetUser).map(_._2).toSet

    //Group games played by top users excluding target user
    val recommendedGames = userAppDetails.filter { case (userId, gameId, _, _) =>
      topUsers.contains(userId) && !gamesByTargetUser.contains(gameId)
    }.groupBy(_._2)

    recommendedGames.foreach { case (gameId, userGames) =>
      val userIds = userGames.map(_._1).mkString(", ") // Comma-separated user IDs
      val gameInfo = userGames.head // Any element contains game details
      println(s"Game ID: $gameId, Title: ${gameInfo._3}, Users: $userIds")
    }
  }

  /**
   * Loads user recommendations from a CSV file
   *
   * @param path The file path to the recommendations file
   * @return A map that returns an array of games recommended by each user
   */
  def loadRecommendations(path: String): Map[Int, Array[Int]] = {
    //Mutable map to store each user's list of appId
    val usersRec = collection.mutable.Map[String, List[Int]]()
    //Read the file and process each line
    Using.resource(Source.fromFile(path)) { source =>
      for (line <- source.getLines().drop(1)) {
        val splitLine = line.split(",")
        val appId = splitLine.head
        val user = splitLine(6)
        // Update the list of app IDs for each user
        usersRec.update(user, appId.toInt :: usersRec.getOrElse(user, Nil))
      }
    }
    //Convert the mutable map to an immutable Map[Int, Array[Int]]
    usersRec.map { case (user, appIds) =>
      (user.toInt, appIds.reverse.toArray)
    }.toMap
  }

  /**
   * Loads games data from a CSV file
   * @param path The file path to the games file
   * @return A map of appId to game titles
   */
  private def loadDataGames(path: String): Map[Int, String] = {
    val gamesRec = collection.mutable.Map[Int, String]()
    //Read the file and process each line
    Using.resource(Source.fromFile(path)) { source =>
      for (line <- source.getLines().drop(1)) {
        val splitLine = line.split(",").map(_.trim)
        val appId = splitLine.head.toInt
        val title = splitLine(1)
        gamesRec.update(appId, title)
      }
    }
    //Convert to an immutable Map[Int, String]
    gamesRec.toMap
  }

  /**
   * Loads game metadata from a JSON file
   *
   * @param path The file path to the JSON file
   * @return A map of appId to arrays of tags
   */
  def loadMetadata(path: String): Map[Int, Array[String]] = {
    //Read the entire JSON file
    val source = Source.fromFile(path)
    implicit val formats: DefaultFormats.type = DefaultFormats
    //Process each line of the JSON file
    val appIdsAndTags = source.getLines().foldLeft(Map.empty[Int, Array[String]]) {
      case (acc, line) =>
        val json = JsonMethods.parse(line)
        val appId = (json \ "app_id").extract[Int]
        val tags = (json \ "tags").extract[Seq[String]].toArray
        acc + (appId -> tags)
    }
    //Close the file after processing
    source.close()
    appIdsAndTags
  }
}
