package com.unibo.recommendationsystem

import com.unibo.recommendationsystem.utils.timeUtils
import org.json4s.jackson.JsonMethods
import org.json4s.{DefaultFormats, Formats}

import scala.io.Source
import scala.math.log
import scala.util.Using
import scala.collection.compat.{toMapViewExtensionMethods, toTraversableLikeExtensionMethods}

class parRecommendation(dataRecPath: String, dataGamesPath: String, metadataPath: String) {

  private var dataRec: Map[Int, Array[Int]] = Map.empty
  private var dataGames: Map[Int, String] = Map.empty
  private var metadata: Map[Int, Array[String]] = Map.empty


  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (explodedData, filteredData, merged) = timeUtils.time(preprocessData(), "Preprocessing Data", "par")
    println("Calculate term frequency and inverse document frequency...")
    val tfidfValues = timeUtils.time(calculateTFIDF(explodedData, filteredData), "Calculating TF-IDF", "par")
    println("Calculate cosine similarity to get similar users...")
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(tfidfValues, targetUser), "Getting Similar Users", "par")
    println("Calculate final recommendation...")
    timeUtils.time(getFinalRecommendations(topUsersSimilarity, targetUser, merged), "Generating Recommendations", "par")
  }


  /**
   * Preprocess data to generate the necessary mappings for recommendation calculations.
   * Outputs user-word pairs, user-to-words map, and full user-item metadata.
   */
  private def preprocessData(): (List[(Int, String)], Map[Int, List[String]], List[(Int, Int, String, Array[String])]) = {

    dataRec = loadRecommendations(dataRecPath)
    dataGames = loadDataGames(dataGamesPath)
    metadata = loadMetadata(metadataPath)

    /*val merged = dataRec.toList.collect {
      case (userId, appIds) =>
        appIds.flatMap { appId =>
          for {
            title <- dataGames.get(appId)
            tags <- metadata.get(appId)
          } yield (userId, appId, title, tags.map(_.trim.toLowerCase.replaceAll("\\s+", " ")))
        }
    }.flatten*/

    val merged = dataRec.flatMap { case (userId, appIds) =>
  appIds.flatMap { appId =>
    for {
      title <- dataGames.get(appId)
      tags <- metadata.get(appId)
    } yield (userId, appId, title, tags.map(_.trim.toLowerCase.replaceAll("\\s+", " ")))
  }
}.toList


    // Step 3: Clean and format merged data
    val cleanMerge = merged.map { case (userId, appId, title, tags) =>
      (userId, appId, title, tags.mkString(","))
    }

    // Step 4: Group by userId and aggregate tags
    /*val filteredData = cleanMerge.groupBy(_._1).map { case (userId, records) =>
      userId -> records.flatMap(_._4.split(",")).distinct
    }*/
    val filteredData = cleanMerge.groupMapReduce(_._1)(_._4.split(",").distinct.toList)(_ ++ _)

    // Step 5: Explode data to get a list of (userId, tag) pairs
    val explodedData = filteredData.flatMap { case (userId, tags) =>
      tags.map(tag => (userId, tag))
    }.toList

    (explodedData, filteredData, merged)
  }

  /**
   * Calculates Term Frequency (TF) and Inverse Document Frequency (IDF) for each word-user pair.
   */
  private def calculateTFIDF(explodedData: List[(Int, String)], filteredData: Map[Int, Seq[String]]): Map[Int, Map[String, Double]] = {
    // Step 1: Calculate Term Frequency (TF)
    val tf = explodedData//.par
      .groupBy { case (userId, word) => (userId, word) }
      .map { case ((userId, word), occurrences) => (userId, word) -> occurrences.size }
      .groupBy { case ((userId, _), _) => userId }
      .map { case (userId, userWordCounts) =>
        val totalWords = userWordCounts.values.sum
        userId -> userWordCounts.map { case ((_, word), count) => word -> (count.toDouble / totalWords) }
      }


    // Step 2: Calculate Document Frequency (DF)
    val wordUserCount = explodedData.map(_._2).distinct.groupBy(identity).map { case (word, occurrences) => word -> occurrences.size }
    val totalDocs = filteredData.size.toDouble

    // Step 3: Calculate TF-IDF
    tf.map { case (userId, userTf) =>
      userId -> userTf.map { case (word, termFreq) =>
        val idf = log(totalDocs / (wordUserCount.getOrElse(word, 1).toDouble))
        word -> (termFreq * idf)
      }
    }
  }

  /**
   * Computes cosine similarity between the target user and all other users.
   */
  private def computeCosineSimilarity(tfidf: Map[Int, Map[String, Double]], targetUser: Int): List[Int] = {
    def cosineSimilarity(v1: Map[String, Double], v2: Map[String, Double]): Double = {
      val dotProduct = v1.keys.map(k => v1.getOrElse(k, 0.0) * v2.getOrElse(k, 0.0)).sum
      val magnitude = math.sqrt(v1.values.map(v => v * v).sum) * math.sqrt(v2.values.map(v => v * v).sum)
      if (magnitude == 0) 0.0 else dotProduct / magnitude
    }

    val targetVector = tfidf.getOrElse(targetUser, Map.empty)
    tfidf.view.filter { case (key, _) => key != targetUser }
      .mapValues(cosineSimilarity(targetVector, _))
      .toList.sortBy(-_._2)
      .take(3)
      .map(_._1)
    /*
    tfidf.view.par
      .filter { case (key, _) => key != targetUser }
      .map { case (key, vector) => key -> cosineSimilarity(targetVector, vector) }
      .toList
      .sortBy(-_._2)
      .take(3)
      .map(_._1)
     */
  }

  /**
   * Extracts final recommendations based on top similar users.
   */
  private def getFinalRecommendations(topUsers: List[Int], targetUser: Int, cleanMerge: List[(Int, Int, String, Array[String])]): Unit = {
    val gamesByTargetUser = cleanMerge.collect { case (`targetUser`, appId, _, _) => appId }.toSet
    cleanMerge.filter { case (userId, gameId, _, _) => topUsers.contains(userId) && !gamesByTargetUser.contains(gameId) }
      .groupBy(_._2)
      .foreach { case (_, userGamePairs) =>
        val gameTitle = userGamePairs.head._3
        val userGroups = userGamePairs.groupBy(_._1) // Group users by their ID
        val groupedUsers = userGroups.mapValues(_.map(_._2)) // Extract game IDs for each user
        groupedUsers.foreach { case (userId, _) =>
          println(s"Recommended game: $gameTitle")
          println(s"  - Recommended by user $userId")
        }
      }
  }

  def loadRecommendations(path: String): Map[Int, Array[Int]] = {
    // Mutable map to store each user's list of app IDs
    val usersRec = collection.mutable.Map[String, List[Int]]()
    // Read the file and process each line
    Using.resource(Source.fromFile(path)) { source =>
      for (line <- source.getLines().drop(1)) {
        val splitLine = line.split(",")
        val appId = splitLine(0)
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
        val appId = splitLine(0).toInt  // Convert appId to Int
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
