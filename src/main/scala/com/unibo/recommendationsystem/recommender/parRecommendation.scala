package com.unibo.recommendationsystem.recommender

import com.unibo.recommendationsystem.utils.timeUtils
import org.json4s.{DefaultFormats, TypeInfo}
import org.json4s.jackson.JsonMethods

import scala.collection.concurrent.TrieMap
import scala.collection.parallel.{ParMap, ParSeq}
import scala.io.Source
import scala.util.Using

class parRecommendation(dataRecPath: String, dataGamesPath: String, metadataPath: String) {
  private val dataRec: Map[Int, Array[Int]] = loadRecommendations(dataRecPath)
  private val dataGames: Map[Int, String] = loadDataGames(dataGamesPath)
  private val metadata: Map[Int, Array[String]] = loadMetadata(metadataPath)

  /**
   * (Parallel version) Generates personalized recommendations for a target user
   *
   * @param targetUser Int, The ID of the user for which we are generating recommendations
   *
   */
  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (userTagsMap, userAppDetails) = timeUtils.time(preprocessData(), "Preprocessing Data", "Par")
    println("Calculate term frequency and inverse document frequency...")
    val tfidfValuesPar = timeUtils.time(calculateTFIDF(userTagsMap), "Calculating TF-IDF", "Par")
    println("Calculate cosine similarity to get similar users...")
    val topUsersSimilarityPar= timeUtils.time(computeCosineSimilarity(tfidfValuesPar, targetUser), "Getting Similar Users", "Par")
    println("Calculate final recommendation...")
    timeUtils.time(generateFinalRecommendations(topUsersSimilarityPar, targetUser, userAppDetails), "Generating Recommendations", "Par")
  }

  /**
   * Preprocesses the input data to create intermediate scala.collections needed for further calculations.
   *
   * @return A tuple of:
   *         - ParSeq[(Int, String)], map each user with his tags for tf-idf calculation
   *         - ParSeq[(Int, Int, String, Array[String])], full user-item metadata
   */
  private def preprocessData(): (ParSeq[(Int, String)], ParSeq[(Int, Int, String, Array[String])]) = {
    //Combines user, game, and metadata details, filtering out empty tags
    val userAppDetails: ParSeq[(Int, Int, String, Array[String])] = dataRec
      .par
      .flatMap { case (userId, appIds) =>
        appIds.flatMap { appId =>
          for {
            title <- dataGames.get(appId)
            tags <- metadata.get(appId)
          } yield (userId, appId, title, tags.map(_.trim.toLowerCase.replaceAll("\\s+", " ")))
        }
      }.filter(_._4.nonEmpty).toSeq

    val userTags: ParSeq[(Int, Int, String, String)] = userAppDetails.map(d =>
      (d._1, d._2, d._3, d._4.mkString(","))
    )

    val groupedUserTags: ParSeq[(Int, Seq[String])] = userTags.map { case (id, _, _, tags) =>
      val cleanedTags = tags.split(",").filter(_.nonEmpty).toSeq
      (id, cleanedTags)
    }.filter(_._2.nonEmpty)

    val userTagsVector: ParSeq[(Int, ParSeq[String])] = groupedUserTags
      .groupBy(_._1)
      .map { case (id, grouped) =>
        val mergedTags = grouped.flatMap(_._2)
        (id, mergedTags)
      }.toSeq

    val explodeUserTag: Seq[(Int, String)] = userTagsVector
      .flatMap { case (userId, tags) => tags.map(tag => (userId, tag)) }.seq

    (explodeUserTag.par, userAppDetails)
  }
  /*
   * userTagsCouples
   * (5382866,strategy)
   * (5382866,classic)
   * (5382866,funny)
   * (5382866,turn-based)
   * (5382866,local multiplayer)
   * (5382866,classic)
   * (5382866,2d)
   * (5382866,retro)
   * (5382866,online co-op)
   * (5382866,multiplayer)
   * (5382866,anime)
   * (5382866,arcade)
   */

  /*
   * userAppDetails
   * (5382866,272270,Torment: Tides of Numenera, [...tags...] )
   * (5382866,343340,Tiamat X, [...tags...])
   * (5382866,233470,Evoland,[...tags...] )
   * (5382866,599880,Tequila Zombies 3, [...tags...] )
   * (5382866,357190,ULTIMATE MARVEL VS. CAPCOM 3, [...tags...])
   * (5382866,544750,SOULCALIBUR VI, [...tags...])
   * (5382866,1659600,Teenage Mutant Ninja Turtles: The Cowabunga Collection, [...tags...])
   * (5382866,483420,Adam Wolfe, [...tags...])
   * (5382866,296770,Real Boxing™, [...tags...])
 */

  /**
   * Computes TF-IDF values for all users based on their tags
   * @param userTagsMap ParSeq[(Int, String)], map each user with his tags for tf-idf calculation
   * @return ParMap[Int, Map[String, Double] ], A map where each user ID maps to another map of tags and their respective TF-IDF scores
   */
  private def calculateTFIDF(userTagsMap: ParSeq[(Int, String)]): ParMap[Int, Map[String, Double]] = {
    //TF
    val calculateTF = (tags: String) => {
      val allTags = tags.split(",")
      allTags.groupBy(identity).mapValues(_.length.toDouble / allTags.length)
    }

    //IDF
    val calculateIDF = (userTagsMap: ParMap[Int, String]) => {
      val userCount = userTagsMap.size
      userTagsMap.values
        .par
        .flatMap(_.split(",").distinct)
        .groupBy(identity)
        .map { case (word, occurrences) => (word, math.log(userCount.toDouble / occurrences.size)) }
    }

    val userTagsMap: ParMap[Int, String] = userTagsMap
      .groupBy(_._1)
      .mapValues(_.map(_._2).mkString(","))

    val idfValuesTag: Map[String, Double] = calculateIDF(userTagsMap).seq

    //TFIDF
    userTagsMap      .map { case (user, words) =>
        val tfValues = calculateTF(words)
        user -> tfValues.map { case (word, tf) => word -> tf * idfValuesTag.getOrElse(word, 0.0) }
      }
  }
  /*
   * tfIDFValues
   * (7785031,Map(multiplayer -> 0.05147607544411006, physics -> 0.2039327116805472, ...))
   * (7614324,Map(turn-based combat -> 0.058421854161816944, funny -> 0.024955196358551045, ...))
   * (5107791,Map(1980s -> 0.05896140840799852, indie -> 0.007796680564226543, ...))
   */


  /**
   * Computes cosine similarity between the target user and all other users
   *
   * @param tfidfUserTags ParMap[Int, Map[String, Double] ], A map where each user ID maps to another map of tags and their respective TF-IDF scores
   * @param targetUser Int, the ID of the target user
   * @return A list of the top 3 most similar user IDs
   */
  private def computeCosineSimilarity(tfidfUserTags: ParMap[Int, Map[String, Double]], targetUser: Int): List[Int] = {

    def cosineSimilarity(targetUserScores: Map[String, Double], otherUserScores: Map[String, Double]): Double = {
      val numerator = targetUserScores.keys.map(k => targetUserScores.getOrElse(k, 0.0) * otherUserScores.getOrElse(k, 0.0)).sum
      val denominator = math.sqrt(targetUserScores.values.map(v => v * v).sum) * math.sqrt(otherUserScores.values.map(v => v * v).sum)
      if (denominator == 0) 0.0 else numerator / denominator
    }

    val targetUserScores = tfidfUserTags(targetUser)

    val topUsers = tfidfUserTags.seq.view
      .filter { case (key, _) => key != targetUser }
      .par
      .map { case (key, otherUserScores) =>
        key -> cosineSimilarity(targetUserScores, otherUserScores)
      }
      .toList           //Convert to list for further processing
      .sortBy(-_._2)    //Sort by similarity score in descending order
      .take(3)          //Take the top 3 most similar users
      .map(_._1)

    topUsers
  }

  /**
   * Generates and prints final game recommendations for a target user based on games played by similar users
   *
   * @param topUsers List[Int], list of IDs of the most similar users
   * @param targetUser Int, the ID of the target user
   * @param userAppDetails ParSeq[(Int, Int, String, Array[String])], full user-item metadata
   */
  private def generateFinalRecommendations(topUsers: List[Int], targetUser: Int, userAppDetails: ParSeq[(Int, Int, String, Array[String])]): Unit = {
    //Get all games played by the target user
    val gamesByTargetUser = userAppDetails.filter(_._1 == targetUser)
      .map(_._2)

    //Group games played by top users excluding target user
    val recommendedGames = userAppDetails.filter {
      case (userId, gameId, _, _) =>
        topUsers.contains(userId) && !gamesByTargetUser.exists(_ == gameId)
    }.groupBy(_._2)

    recommendedGames.foreach { case (gameId, userGames) =>
      val userIds = userGames.map(_._1).mkString(", ")
      val gameInfo = userGames.head
      println(s"Game ID: $gameId, Title: ${gameInfo._3}, Users: $userIds")
    }
  }

  /**
   * Loads user recommendations from a CSV file
   *
   * @param path The file path to the recommendations file
   * @return A map of an array of games recommended by each user
   */
  def loadRecommendations(path: String): Map[Int, Array[Int]] = {
    Using.resource(Source.fromFile(path)) { source =>
      val lines = source.getLines().drop(1).toArray.par
      //Mutable map to store each user's list of appId
      val usersRec = TrieMap[String, List[Int]]()

      lines.foreach { line =>
        val splitLine = line.split(",")
        val appId = splitLine.head
        val user = splitLine(6)

        usersRec.synchronized {
          val updatedList = usersRec.getOrElse(user, List()) :+ appId.toInt
          usersRec.put(user, updatedList)
        }
      }
      //Convert the mutable map to an immutable
      usersRec.map { case (user, appIds) =>
        (user.toInt, appIds.reverse.toArray)
      }.toMap
    }
  }

  /**
   * Loads games data from a CSV file
   *
   * @param path The file path to the games file
   * @return A map of appId to game titles
   */
  private def loadDataGames(path: String): Map[Int, String] = {
    val lines = Using.resource(Source.fromFile(path)) { source =>
      source.getLines().drop(1).toSeq.par
    }

    //Convert to an immutable Map[Int, String]
    val gamesRec = lines.map { line =>
      val splitLine = line.split(",").map(_.trim)
      val appId = splitLine.head.toInt
      val title = splitLine(1)
      appId -> title
    }.toMap

    gamesRec.seq
  }

  /**
   * Loads game metadata from a JSON file
   *
   * @param path The file path to the JSON file
   * @return A map of appId to arrays of tags
   */
  def loadMetadata(path: String): Map[Int, Array[String]] = {
    val source = Source.fromFile(path)
    implicit val formats: DefaultFormats.type = DefaultFormats
    val appIdsAndTags = source.getLines().foldLeft(Map.empty[Int, Array[String]]) {
      case (acc, line) =>
        val json = JsonMethods.parse(line)
        val appId = (json \ "app_id").extract[Int]
        val tags = (json \ "tags").extract[Seq[String]].toArray
        acc + (appId -> tags)
    }
    source.close()
    appIdsAndTags
  }

}