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
   * @param targetUser The user ID for whom the recommendations are being generated.
   */
  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (userTagsMap, merged) = timeUtils.time(preprocessData(), "Preprocessing Data", "Seq")
    println("Calculate term frequency and inverse document frequency...")
    val tfidfValues = timeUtils.time(calculateTFIDF(userTagsMap), "Calculating TF-IDF", "Seq")
    println("Calculate cosine similarity to get similar users...")
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(tfidfValues, targetUser), "Getting Similar Users", "Seq")
    println("Calculate final recommendation...")
    timeUtils.time(generateFinalRecommendations(topUsersSimilarity, targetUser, merged), "Generating Recommendations", "Seq")
  }

  /**
   * Preprocesses the input data to create intermediate RDDs needed for further calculations.
   *
   * @return A tuple of:
   *         - List[(userId, tag)]: Exploded user-tag pairs
   *         - List[(userId, appId, appTitle, appTags)]: Full user-item metadata
   */
  private def preprocessData(): (Map[Int, String], List[(Int, Int, String, Array[String])]) = {

    //Combine user, game, and metadata into a detailed user-app mapping, only includes records where tags are non-empty
      val userAppDetails = dataRec.flatMap {
        case (userId, appIds) =>
          for {
            appId <- appIds
            appTitle <- dataGames.get(appId)
            appTags <- metadata.get(appId)
          } yield (userId, appId, appTitle, appTags.map(_.trim.toLowerCase.replaceAll("\\s+", " ")))
      }.filter(_._4.nonEmpty)
    /*(8897055,616560,Ultimate Epic Battle Simulator,[Ljava.lang.String;@60fe75f7)
    * */


    val userTags = userAppDetails.map {
      case (userId, _, _, appTags) =>
        (userId, appTags.mkString(",").split(",").filter(_.nonEmpty).toList)
    }.filter(_._2.nonEmpty)
    /*(8897055,List(simulation, sandbox, war, strategy, action, funny, medieval, violent, singleplayer, gore, indie, early access, fantasy, open world, zombies, adventure, survival, memes, third person, atmospheric))
    */

    val groupedUserTags: Map[Int, List[String]] = userTags.groupMapReduce(userId => userId._1)(data => data._2)((tags1, tags2) => tags1 ++ tags2)
    //(8897055,List(simulation, sandbox, war, strategy, action, funny, medieval, violent, singleplayer, gore, indie, early access, fantasy, open world, zombies, adventure, survival, memes, third person, atmospheric, visual novel, dating sim, casual, sexual content, cute, 2d, anime, story rich, choices matter, romance, multiple endings, demons, mature, comedy, singleplayer, adventure, nudity, hentai, shooter, psychological horror, rpg, action, adventure, anime, sexual content, visual novel, female protagonist, nudity, cute, jrpg, singleplayer, naval, world war ii, third-person shooter, military, fps, shooter, hentai, naval combat, mature, sexual content, visual novel, casual, anime, adventure, simulation, nudity, memes, comedy, nsfw))

    val explodeUserTag: List[(Int, String)] = groupedUserTags.toList
      .flatMap { case (userId, tags) => tags.map(tag => (userId, tag)) }

    /*
    (8897055,simulation)
    (8897055,sandbox)
    (8897055,war)
    (8897055,strategy)
    (8897055,action)
     */

    val userTagsMap: Map[Int, String] = explodeUserTag
      .groupBy(_._1) // Group by user ID
      .mapValues(_.map(_._2).mkString(",")) // Concatenate all strings for each user


    (userTagsMap, userAppDetails.toList)
  }

  private def calculateTFIDF(userTagsMap: Map[Int, String]): Map[Int, Map[String, Double]] = {
    val calculateTF = (tags: String) => {
      val allTags = tags.split(",")
      allTags.groupBy(identity).mapValues(_.length.toDouble / allTags.length)
    }

    val calculateIDF = (userTagsMap: Map[Int, String]) => {
      val userCount = userTagsMap.size
      userTagsMap.values
        .flatMap(_.split(",").distinct)
        .groupBy(identity)
        .map { case (tag, occurrences) => (tag, math.log(userCount.toDouble / occurrences.size)) }
    }


    val idfValuesTag: Map[String, Double] = calculateIDF(userTagsMap)

    userTagsMap
      .map { case (user, words) =>
        val tfValues = calculateTF(words)
        user -> tfValues.map { case (word, tf) => word -> tf * idfValuesTag.getOrElse(word, 0.0) }
      }

  }

  /**
   * Computes cosine similarity between the target user and all other users.
   */
  private def computeCosineSimilarity(tfidf: Map[Int, Map[String, Double]], targetUser: Int): List[Int] = {

    def cosineSimilarity(v1: Map[String, Double], v2: Map[String, Double], targetMagnitude: Double): Double = {
      val dotProduct = v1.keySet.intersect(v2.keySet).foldLeft(0.0) { (acc, k) => acc + v1(k) * v2(k) }
      val magnitude = targetMagnitude * math.sqrt(v2.values.map(v => v * v).sum)
      if (magnitude == 0) 0.0 else dotProduct / magnitude
    }

    val targetVector = tfidf(targetUser)
    val targetMagnitude = math.sqrt(targetVector.values.map(v => v * v).sum)

    val nonZeroTfidf = tfidf.filter { case (_, vector) => math.sqrt(vector.values.map(v => v * v).sum) != 0 }

    val topUsers = nonZeroTfidf.view
      .filterKeys(_ != targetUser)
      .mapValues(cosineSimilarity(targetVector, _, targetMagnitude))
      .toList.sortBy(-_._2)
      .take(3)
      .map(_._1)

    topUsers
  }

  /**
   * Extracts final recommendations based on top similar users.
   */
  private def generateFinalRecommendations(topUsers: List[Int], targetUser: Int, cleanMerge: List[(Int, Int, String, Array[String])]): Unit = {
    // Step 1: Get all games played by the target user
    val gamesByTargetUser = cleanMerge.filter(_._1 == targetUser).map(_._2).toSet // Convert to Set for faster membership checks

    // Step 2: Group games played by top users (excluding target user)
    val filteredGamesGrouped = cleanMerge.filter { case (userId, gameId, _, _) =>
      topUsers.contains(userId) && !gamesByTargetUser.contains(gameId)
    }.groupBy(_._2) // Group by game ID

    // Step 3: Extract game information and format output
    filteredGamesGrouped.foreach { case (gameId, userGames) =>
      val userIds = userGames.map(_._1).mkString(", ") // Comma-separated user IDs
      val gameInfo = userGames.head // Any element contains game details
      println(s"Game ID: $gameId, Title: ${gameInfo._3}, Users: $userIds")
    }
  }

  def loadRecommendations(path: String): Map[Int, Array[Int]] = {
    // Mutable map to store each user's list of app IDs
    val usersRec = collection.mutable.Map[String, List[Int]]()
    // Read the file and process each line
    Using.resource(Source.fromFile(path)) { source =>
      for (line <- source.getLines().drop(1)) {
        val splitLine = line.split(",")
        val appId = splitLine.head
        val user = splitLine(6)
        // Update the list of app IDs for each user
        usersRec.update(user, appId.toInt :: usersRec.getOrElse(user, Nil))
      }
    }
    // Convert the map to a List[(Int, List[Int])] with user IDs and their app ID lists
    usersRec.map { case (user, appIds) =>
      (user.toInt, appIds.reverse.toArray)  // Convert user ID to Int and list to Array[String]
    }.toMap
  }


  /** Load game data from CSV */
  private def loadDataGames(path: String): Map[Int, String] = {
    val gamesRec = collection.mutable.Map[Int, String]()
    // Read the file and process each line
    Using.resource(Source.fromFile(path)) { source =>
      for (line <- source.getLines().drop(1)) {  // Skip the header line if present
        val splitLine = line.split(",").map(_.trim)  // Split by comma and trim whitespace
        val appId = splitLine.head.toInt  // Convert appId to Int
        val title = splitLine(1)        // Title is the second column
        gamesRec.update(appId, title)
      }
    }
    // Convert to an immutable Map[Int, String]
    gamesRec.toMap
  }

  def loadMetadata(path: String): Map[Int, Array[String]] = {
    // Read the entire JSON file
    val source = Source.fromFile(path)
    implicit val formats: DefaultFormats.type = DefaultFormats

    val appIdsAndTags = source.getLines().foldLeft(Map.empty[Int, Array[String]]) {
      case (acc, line) =>
        val json = JsonMethods.parse(line)
        val appId = (json \ "app_id").extract[Int]
        val tags = (json \ "tags").extract[Seq[String]].toArray  // Convert tags to Array[String]
        acc + (appId -> tags)  // Add appId and tags to the accumulator map
    }
    source.close()
    appIdsAndTags
  }







}
