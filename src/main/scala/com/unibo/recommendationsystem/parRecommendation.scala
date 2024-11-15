package com.unibo.recommendationsystem

import com.unibo.recommendationsystem.utils.timeUtils
import org.json4s.jackson.JsonMethods
import org.json4s.{DefaultFormats, Formats}

import scala.io.Source
import scala.math.{exp, log}
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

    tfidfValues.get(4893896) match {
      case Some(userTfIdf) =>
        println(s"TF-IDF for user 4893896:")
        userTfIdf.foreach { case (word, tfidfValue) =>
          println(s"Word: $word, TF-IDF: $tfidfValue")
        }
      case None =>
        println("User 4893896 not found in the TF-IDF results.")
    }

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

    val cleanedData = cleanMerge.map { case (id, _, _, tags) =>
      val cleanedTags = tags.split(",").filter(_.nonEmpty).toList
      (id, cleanedTags)
    }.filter(_._2.nonEmpty)

    /*val extractedRecords = cleanedData.filter { case (id, _) => id == 4893896 }

    // Print each list of tags as-is
    extractedRecords.foreach { case (_, tags) =>
      println(tags)
    }
    println("list cleanedData par")

     */

    val filteredData = cleanedData.groupMapReduce(_._1)(_._2)(_ ++ _)

    //List for 4893896: List(strategy, rts, medieval, multiplayer, classic, historical, base building, city builder, resource management, singleplayer, tactical, real-time, co-op, competitive, replay value, remake, 2d, isometric, action, adventure, horror, online co-op, survival horror, co-op, multiplayer, psychological horror, vr, pve, survival, asymmetric vr, action-adventure, atmospheric, dark, pvp, exploration, first-person, adventure, action, 3d, early access, fps, war, realistic, military, multiplayer, shooter, tactical, action, first-person, historical, team-based, simulation, gore, cold war, atmospheric, singleplayer, massively multiplayer, violent, strategy, indie, strategy, world war ii, turn-based strategy, military, turn-based, hex grid, wargame, grand strategy, historical, management, war, turn-based tactics, tactical, artificial intelligence, multiplayer, singleplayer, turn-based combat, replay value, simulation, 3d, strategy, world war ii, war, simulation, rts, action, tactical, military, multiplayer, real time tactics, realistic, tanks, historical, co-op, singleplayer, destruction, atmospheric, difficult, classic, adventure, open world, looter shooter, multiplayer, third-person shooter, action, co-op, shooter, rpg, online co-op, post-apocalyptic, survival, massively multiplayer, tactical, third person, singleplayer, mmorpg, adventure, atmospheric, fps, stealth, strategy, world war ii, turn-based strategy, turn-based, hex grid, turn-based tactics, indie, turn-based combat, world war ii, action, fps, realistic, multiplayer, singleplayer, shooter, war, tactical, military, simulation, historical, team-based, classic, strategy, atmospheric, first-person, tanks, survival, world war ii, action, fps, multiplayer, co-op, shooter, first-person, violent, stealth, heist, adventure, gore, rpg, singleplayer, comedy, great soundtrack, psychological horror, memes, atmospheric, online co-op, rpg, indie, metroidvania, platformer, action, comedy, 2d, exploration, fantasy, adventure, co-op, singleplayer, multiplayer, funny, side scroller, pixel graphics, retro, magic, epic, online co-op, world war ii, fps, multiplayer, war, realistic, action, shooter, military, tactical, singleplayer, first-person, strategy, historical, co-op, gore, indie, simulation, online co-op, open world, early access, horror, free to play, cute, first-person, singleplayer, indie, atmospheric, psychological horror, cartoon, dark humor, funny, dark, adventure, procedural generation, survival horror, survival, action, 3d, walking simulator, comedy, action, memes, survival, zombies, co-op, base building, multiplayer, adventure, stealth, open world, crafting, strategy, open world survival craft, singleplayer, comedy, third person, tower defense, psychological horror, horror, violent, rpg, action rpg, hack and slash, dungeon crawler, fantasy, singleplayer, loot, indie, moddable, action, adventure, isometric, magic, cartoon, exploration, top-down, female protagonist, steampunk, great soundtrack, co-op, free to play, hunting, multiplayer, open world, simulation, shooter, co-op, first-person, survival, realistic, online co-op, adventure, fps, sports, singleplayer, action, stealth, strategy, massively multiplayer, casual, strategy, world war ii, tactical, war, rts, multiplayer, military, realistic, real time tactics, action, co-op, historical, simulation, moddable, tanks, destruction, singleplayer, level editor, epic, sandbox, naval combat, simulation, submarine, world war ii, naval, military, historical, underwater, open world, realistic, singleplayer, action, multiplayer, strategy, classic, war, strategy, world war ii, turn-based strategy, wargame, turn-based, hex grid, historical, tanks, singleplayer, tactical, military, grand strategy, multiplayer, war, replay value, turn-based tactics, fps, zombies, co-op, survival, horror, action, multiplayer, online co-op, gore, shooter, team-based, first-person, survival horror, moddable, great soundtrack, class-based, difficult, singleplayer, comedy, adventure, shooter, co-op, pvp, massively multiplayer, world war i, indie, action, fps, first-person, historical, strategy, multiplayer, wargame, war, atmospheric, military, simulation, realistic, gore, tactical, world war ii, simulation, strategy, management, flight, singleplayer, indie, war, difficult, survival, action, casual, 3d, military, cartoony, roguelike, perma death, roguelite, cute, multiplayer, strategy, free to play, world war ii, simulation, turn-based, turn-based strategy, wargame, multiplayer, hex grid, singleplayer, war, turn-based tactics, tactical, historical, replay value, turn-based combat, asynchronous multiplayer, co-op, moddable, level editor, action, simulation, fps, shooter, tactical, military, realistic, open world, world war ii, war, military, tanks, first-person, realistic, simulation, fps, shooter, tactical, third person, action, strategy, historical, multiplayer, singleplayer, open world, trackir, sandbox, atmospheric, indie, surreal, simulation, memes, clicker, singleplayer, 2d, indie, action, great soundtrack, casual, strategy, atmospheric, sports, life sim, relaxing, story rich, family friendly, adventure, lore-rich, character customization, stylized, multiplayer, flight, side scroller, shoot 'em up, football (soccer), strategy, pvp, competitive, shooter, arcade, 2d fighter, 2d, funny, controller, casual, team-based, vehicular combat, simulation, action, cartoony)


    // Step 5: Explode data to get a list of (userId, tag) pairs
    /*val explodedData = filteredData.flatMap { case (userId, tags) =>
      tags.map(tag => (userId, tag))
    }.toList
     */

    val explodedData: List[(Int, String)] = filteredData.toList
      .flatMap { case (userId, tags) => tags.map(tag => (userId, tag)) }

    // Print the result to check all entries are included
   // println(explodedData)


   // val test = explodedData.filter { case (userId, _) => userId == 4893896 }
//
    //println(test)

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

    val targetVector = tfidf(targetUser) // Get the vector for the target user

    val topUsers = tfidf.view
      .filter { case (key, _) => key != targetUser } // Exclude the target user itself
      .mapValues(cosineSimilarity(targetVector, _)) // Calculate cosine similarity between targetVector and each other user's vector
      .toList // Convert to a list for sorting
      .sortBy(-_._2) // Sort by similarity score in descending order
      .take(3) // Take the top 3 most similar users
      .map(_._1) // Extract the user keys (IDs)
    /*
    tfidf.view.par
      .filter { case (key, _) => key != targetUser }
      .map { case (key, vector) => key -> cosineSimilarity(targetVector, vector) }
      .toList
      .sortBy(-_._2)
      .take(3)
      .map(_._1)
     */
    topUsers
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
