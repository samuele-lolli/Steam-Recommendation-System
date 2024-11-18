package com.unibo.recommendationsystem.recommender

import com.unibo.recommendationsystem.utils.timeUtils
import org.json4s.DefaultFormats
import org.json4s.jackson.JsonMethods

import scala.collection.concurrent.TrieMap
import scala.collection.parallel.{ParMap, ParSeq}
import scala.io.Source
import scala.util.Using

class parRecommendation(dataRecPath: String, dataGamesPath: String, metadataPath: String) {


  // Load data
  private val dataRec: Map[Int, Array[Int]] = loadRecommendations(dataRecPath)
  private val dataGames: Map[Int, String] = loadDataGames(dataGamesPath)
  private val metadata: Map[Int, Array[String]] = loadMetadata(metadataPath)


  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (explodedData, merged) = timeUtils.time(preprocessData(), "Preprocessing Data", "Par")
    println("Calculate term frequency and inverse document frequency...")
    val tfidfValuesPar = timeUtils.time(calculateTFIDF(explodedData), "Calculating TF-IDF", "Par")
      println("Calculate cosine similarity to get similar users...")
    val topUsersSimilarityPar= timeUtils.time(computeCosineSimilarity(tfidfValuesPar, targetUser), "Getting Similar Users", "Par")
    println("Calculate final recommendation...")
    timeUtils.time(getFinalRecommendations(topUsersSimilarityPar, targetUser, merged), "Generating Recommendations", "Par")
  }


  private def preprocessData(): (ParSeq[(Int, String)], ParSeq[(Int, Int, String, Array[String])]) = {

    // Utilizza un ParSeq per memorizzare i dettagli dell'app dell'utente
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

    // Step 3: Pulisci e formatta i dati uniti
    val cleanMerge: ParSeq[(Int, Int, String, String)] = userAppDetails.map(d =>
      (d._1, d._2, d._3, d._4.mkString(","))
    )

   val cleanedData: ParSeq[(Int, Seq[String])] = cleanMerge.map { case (id, _, _, tags) =>
     val cleanedTags = tags.split(",").filter(_.nonEmpty).toSeq // Use Seq instead of ParArray
     (id, cleanedTags)
   }.filter(_._2.nonEmpty)

    // Aggregate data without `groupMapReduce`
    val filteredData: ParSeq[(Int, ParSeq[String])] = cleanedData
      .groupBy(_._1) // Group by userId
      .map { case (id, grouped) =>
        val mergedTags = grouped.flatMap(_._2) // Flatten tags
        (id, mergedTags)
      }.toSeq

    // Utilizza `flatMap` per esplodere i dati
    val explodedData: Seq[(Int, String)] = filteredData
      .flatMap { case (userId, tags) => tags.map(tag => (userId, tag)) }.seq

    (explodedData.par, userAppDetails)
  }


  private def calculateTFIDF(explodedList: ParSeq[(Int, String)]): ParMap[Int, Map[String, Double]] = {
    val calculateTF = (userWords: String) => {
      val wordsSplit = userWords.split(",")
      wordsSplit.groupBy(identity).mapValues(_.length.toDouble / wordsSplit.length)
    }

    val calculateIDF = (groupedWords: ParMap[Int, String]) => {
      val userCount = groupedWords.size
      groupedWords.values
        .par
        .flatMap(_.split(",").distinct)
        .groupBy(identity)
        .map { case (word, occurrences) => (word, math.log(userCount.toDouble / occurrences.size)) }
    }

    val groupedUserWords: ParMap[Int, String] = explodedList
      .groupBy(_._1) // Group by user ID
      .mapValues(_.map(_._2).mkString(",")) // Concatenate all strings for each user

    val idfValues: Map[String, Double] = calculateIDF(groupedUserWords).seq.toMap

    groupedUserWords
      .map { case (user, words) =>
        val tfValues = calculateTF(words)
        user -> tfValues.map { case (word, tf) => word -> tf * idfValues.getOrElse(word, 0.0) }
      }
  }

  private def computeCosineSimilarity(tfidf: ParMap[Int, Map[String, Double]], targetUser: Int): List[Int] = {

  def cosineSimilarity(v1: Map[String, Double], v2: Map[String, Double]): Double = {
    val dotProduct = v1.keys.map(k => v1.getOrElse(k, 0.0) * v2.getOrElse(k, 0.0)).sum
    val magnitude = math.sqrt(v1.values.map(v => v * v).sum) * math.sqrt(v2.values.map(v => v * v).sum)
    if (magnitude == 0) 0.0 else dotProduct / magnitude
  }

  val targetVector = tfidf(targetUser) // Get the vector for the target user
  val topUsers = tfidf.seq.view
    .filter { case (key, _) => key != targetUser }
    .par // Exclude the target user itself
    .map { case (key, vector) =>
      key -> cosineSimilarity(targetVector, vector) // Calculate cosine similarity
    }
    .toList // Convert to list for further processing
    .sortBy(-_._2) // Sort by similarity score in descending order
    .take(3) // Take the top 3 most similar users
    .map(_._1)
  // Extract the user keys (IDs)
  topUsers
}

  private def getFinalRecommendations(topUsers: List[Int], targetUser: Int, cleanMerge: ParSeq[(Int, Int, String, Array[String])]): Unit = {
    // Converti cleanMerge in una collezione parallela
    // Step 1: Ottieni tutti i giochi giocati dall'utente target
    val gamesByTargetUser = cleanMerge.filter(_._1 == targetUser)
      .map(_._2)

    // Step 2: Filtra cleanMerge per ottenere i giochi giocati dai top utenti ma non dall'utente target
    val filteredGamesGrouped = cleanMerge.filter {
      case (userId, gameId, _, _) =>
        topUsers.contains(userId) && !gamesByTargetUser.exists(_ == gameId)
    }.groupBy(_._2)

   filteredGamesGrouped.foreach { case (gameId, userGames) =>
     val userIds = userGames.map(_._1).mkString(", ") // Comma-separated user IDs
     val gameInfo = userGames.head // Any element contains game details
     println(s"Game ID: $gameId, Title: ${gameInfo._3}, Users: $userIds")
   }
  }


  def loadRecommendations(path: String): Map[Int, Array[Int]] = {
    // Use Using to safely manage file resources
    Using.resource(Source.fromFile(path)) { source =>
      // Read lines, drop the header, and convert to a ParArray
      val lines = source.getLines().drop(1).toArray.par

      // Use a concurrent map to safely aggregate results in parallel
      val usersRec = TrieMap[String, List[Int]]()

      // Process lines in parallel
      lines.foreach { line =>
        val splitLine = line.split(",")
        val appId = splitLine.head
        val user = splitLine(6)

        // Update the TrieMap manually
        usersRec.synchronized {
          val updatedList = usersRec.getOrElse(user, List()) :+ appId.toInt
          usersRec.put(user, updatedList)
        }
      }

      // Convert the TrieMap to an immutable Map
      usersRec.map { case (user, appIds) =>
        (user.toInt, appIds.reverse.toArray)
      }.toMap
    }
  }

  /** Load game data from CSV */
  private def loadDataGames(path: String): Map[Int, String] = {
    // Read the file into memory
    val lines = Using.resource(Source.fromFile(path)) { source =>
      source.getLines().drop(1).toSeq.par // Skip header and load lines into a Seq
    }

    // Process lines in parallel to create a Map[Int, String]
    val gamesRec = lines.map { line =>
      val splitLine = line.split(",").map(_.trim) // Split by comma and trim whitespace
      val appId = splitLine.head.toInt // Convert appId to Int
      val title = splitLine(1) // Title is the second column
      appId -> title // Return a tuple (appId, title)
    }.toMap // Combine results into an immutable Map

    gamesRec.seq
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