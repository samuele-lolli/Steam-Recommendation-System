package com.unibo.recommendationsystem

import com.unibo.recommendationsystem.utils.timeUtils
import org.json4s.jackson.JsonMethods
import org.json4s.DefaultFormats

import scala.io.Source
import scala.util.Using
import scala.collection.compat.{toMapViewExtensionMethods, toTraversableLikeExtensionMethods}

class seqRecommendation(dataRecPath: String, dataGamesPath: String, metadataPath: String) {

  private var dataRec: Map[Int, Array[Int]] = Map.empty
  private var dataGames: Map[Int, String] = Map.empty
  private var metadata: Map[Int, Array[String]] = Map.empty


  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (explodedData, merged) = timeUtils.time(preprocessData(), "Preprocessing Data", "par")
    println("Calculate term frequency and inverse document frequency...")
    val tfidfValues = timeUtils.time(calculateTFIDF(explodedData), "Calculating TF-IDF", "par")
    println("Calculate cosine similarity to get similar users...")
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(tfidfValues, targetUser), "Getting Similar Users", "par")
    println("Calculate final recommendation...")
    timeUtils.time(getFinalRecommendations(topUsersSimilarity, targetUser, merged), "Generating Recommendations", "par")
  }


  /**
   * Preprocess data to generate the necessary mappings for recommendation calculations.
   * Outputs user-word pairs, user-to-words map, and full user-item metadata.
   */
  private def preprocessData(): (List[(Int, String)], List[(Int, Int, String, Array[String])]) = {

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
    }.toList.filter(_._4.nonEmpty)

    // Step 3: Clean and format merged data
    val cleanMerge = merged.map { case (userId, appId, title, tags) =>
      (userId, appId, title, tags.mkString(","))
    }

    val cleanedData = cleanMerge.map { case (id, _, _, tags) =>
      val cleanedTags = tags.split(",").filter(_.nonEmpty).toList
      (id, cleanedTags)
    }.filter(_._2.nonEmpty)

    val filteredData = cleanedData.groupMapReduce(_._1)(_._2)(_ ++ _)

    val explodedData: List[(Int, String)] = filteredData.toList
      .flatMap { case (userId, tags) => tags.map(tag => (userId, tag)) }


    (explodedData, merged)
  }

  /**
   * Calculates Term Frequency (TF) and Inverse Document Frequency (IDF) for each word-user pair.
   */
  private def calculateTFIDF(explodedList: List[(Int, String)]): Map[Int, Map[String, Double]] = {
    val calculateTF = (userWords: String) => {
      val wordsSplit = userWords.split(",")
      wordsSplit.groupBy(identity).mapValues(_.length.toDouble / wordsSplit.length)
    }

    val calculateIDF = (groupedWords: Map[Int, String]) => {
      val userCount = groupedWords.size
      groupedWords.values
        .flatMap(_.split(",").distinct)
        .groupBy(identity)
        .map { case (word, occurrences) => (word, math.log(userCount.toDouble / occurrences.size)) }
    }

    val groupedUserWords: Map[Int, String] = explodedList
      .groupBy(_._1) // Group by user ID
      .mapValues(_.map(_._2).mkString(",")) // Concatenate all strings for each user

    val idfValues: Map[String, Double] = calculateIDF(groupedUserWords)

    groupedUserWords.map { case (user, words) =>
      val tfValues = calculateTF(words)
      user -> tfValues.map { case (word, tf) => word -> tf * idfValues.getOrElse(word, 0.0) }
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

    val targetVector = tfidf(targetUser) // Get the vector for the target user

    val topUsers = tfidf.view
      .filter { case (key, _) => key != targetUser } // Exclude the target user itself
      .mapValues(cosineSimilarity(targetVector, _)) // Calculate cosine similarity between targetVector and each other user's vector
      .toList // Convert to a list for sorting
      .sortBy(-_._2) // Sort by similarity score in descending order
      .take(3) // Take the top 3 most similar users
      .map(_._1) // Extract the user keys (IDs)

    topUsers
  }

  /**
   * Extracts final recommendations based on top similar users.
   */
  private def getFinalRecommendations(topUsers: List[Int], targetUser: Int, cleanMerge: List[(Int, Int, String, Array[String])]): Unit = {
    // Step 1: Get all games played by the target user
    val gamesByTargetUser = cleanMerge.filter(_._1 == targetUser)
      .map(_._2)
    // Step 2: Map to extract appId
    gamesByTargetUser.foreach(println)
    println("All user games par")

    // Step 2: Filter cleanMerge to get games played by topUsers but not by the target user
    val filteredGames = cleanMerge.filter {
      case (userId, gameId, _, _) =>
        topUsers.contains(userId) && !gamesByTargetUser.contains(gameId)
    }

    val extractedGames = filteredGames.map {
      case (userId, gameId, title, _) => (userId, gameId, title)
    }

    // Step 2: Print each extracted tuple
    extractedGames.foreach { case (userId, gameId, title) =>
      println(s"User ID: $userId, Game ID: $gameId, Title: $title")
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


  /*def loadRecommendations(path: String): Map[Int, Array[Int]] = {
    // Read the file and process in parallel
    val usersRec = collection.parallel.mutable.ParMap[String, List[Int]]()

    // Read the file lazily to reduce memory consumption
    Using.resource(Source.fromFile(path)) { source =>
      source.getLines().drop(1).grouped(10000).foreach { linesBatch =>
        linesBatch.par.foreach { line =>
          val splitLine = line.split(",")
          val appId = splitLine(0)
          val user = splitLine(6)
          // Update the list of app IDs for each user
          usersRec.updated(user, appId.toInt :: usersRec.getOrElse(user, Nil))
        }
      }
    }

    // Convert the parallel map to an immutable map
    usersRec.map { case (user, appIds) =>
      (user.toInt, appIds.reverse.toArray)
    }.seq.toMap // Convert to sequential collection before returning
  }

   */



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


 /* def loadMetadata(path: String): Map[Int, Array[String]] = {
    implicit val formats: DefaultFormats.type = DefaultFormats // JSON deserialization formats

    // Read the file and process each line in parallel
    val source = Source.fromFile(path)
    try {
      source.getLines()
        .toSeq // Convert to a sequential collection
        .par // Convert to a parallel collection
        .map { line =>
          val json = JsonMethods.parse(line) // Parse the JSON line
          val appId = (json \ "app_id").extract[Int] // Extract appId
          val tags = (json \ "tags").extract[Seq[String]].toArray // Extract tags as Array[String]
          appId -> tags // Return as a tuple (appId, tags)
        }
        //.seq
        .toMap // Convert the parallel collection to a map
    } finally {
      source.close() // Ensure the file is closed
    }
  }*/




}
