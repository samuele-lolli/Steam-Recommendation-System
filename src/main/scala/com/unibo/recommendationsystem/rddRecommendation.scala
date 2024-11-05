package com.unibo.recommendationsystem

import com.unibo.recommendationsystem.utils.timeUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.apache.spark.sql.execution.streaming.CommitMetadata.format

import scala.reflect.ClassTag

class rddRecommendation(spark: SparkSession, datapathRec: String, datapathGames: String, datapathMetadata: String) {

  private def removeHeader(rdd: RDD[String]): RDD[String] = rdd.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }

  private val rddRec = removeHeader(spark.sparkContext.textFile(datapathRec))

  private val rddGames = removeHeader(spark.sparkContext.textFile(datapathGames))

  private val rddMetadata: RDD[String] = spark.sparkContext.textFile(datapathMetadata)

  // Logica di raccomandazione
  def recommend(targetUser: Int): Unit = {
    // Time the preprocessing of data
    println("Preprocessing data...")
    val (mergedRdd, explodedRDD, gamesData) = timeUtils.time(preprocessData(), "Preprocessing Data")
    /*
    Elapsed time for Preprocessing Data:	89ms (89484250ns)
    */
    // Time the TF-IDF calculation
    println("Calculate term frequency and inverse document frequency...")
    val tfidfValues = timeUtils.time(calculateTFIDF(explodedRDD), "Calculating TF-IDF")
    /*
    Elapsed time for Calculating TF-IDF:	181639ms (181639066875ns)
     */

    // Time the similarity computation
    println("Calculate cosine similarity to get similar users...")
    val topUsersSimilarity = timeUtils.time(getSimilarUsers(targetUser, tfidfValues), "Getting Similar Users")
    /*
    Elapsed time for Getting Similar Users:	204060ms (204060907125ns)
    */
    // Time the recommendation generation
    println("Calculate final recommendation...")
    timeUtils.time(getRecommendation(mergedRdd, topUsersSimilarity, gamesData, targetUser), "Generating Recommendations")
    /*
    Elapsed time for Generating Recommendations:	41423ms (41423732625ns)
     */
    //Total time of execution: 427211ms

  }

  // Preprocessing dei dati
  private def preprocessData(): (RDD[(Int, String, String)], RDD[(String, String)], RDD[(Int, String)]) = {

    case class recommend(appId: Int, userId: String, isRecommended: Boolean)

    val appUserRDD = rddRec.map{ row =>
      val fields = row.split(",").map(_.trim)
      recommend(fields(0).toInt, fields(6), fields(4).toBoolean)
    }

    // Funzione per estrarre `app_id` e `user_id` da `rddRec`
    /* val extractAppIdUserId: String => (Int, String, Boolean) = line => {
       val fields = line.split(",")
       val appId = fields(0).toInt
       val userId = fields(6)
       val isRecommended = fields(4).toBoolean
       (appId, userId, isRecommended)
     }

     */

    // Processa `rddRec` per ottenere un RDD di (appId, userId)
    //val appUserRDD: RDD[(Int, String, Boolean)] = rddRec.map(extractAppIdUserId)

    val recommendedRDD: RDD[(Int, String)] = appUserRDD
      .filter(_.isRecommended)                    // Filter only recommended items
      .map(recommend => (recommend.appId, recommend.userId))

    // Funzione per estrarre `app_id` e `tags` da `rddMetadata`
    val extractAppIdTags: String => (Int, String) = line => {
      val json = parse(line)
      val appId = (json \ "app_id").extract[Int]
      val tagsArray = (json \ "tags").extract[Array[String]]
      val tags = tagsArray.mkString(",")
      (appId, tags)
    }

    val tagsRDD: RDD[(Int, String)] = rddMetadata.map(extractAppIdTags)



    case class AppTags(appId: Int, tags: String)

    // Define the function to parse JSON and create an AppTags instance
    /*val extractAppIdTags: String => AppTags = line => {
      val json = parse(line)
      val appId = (json \ "app_id").extract[Int]
      val tagsArray = (json \ "tags").extract[Array[String]]
      val tags = tagsArray.mkString(",")
      AppTags(appId, tags)
    }*/



    // Map each line in rddMetadata to an AppTags instance
    //val tagsRDD: RDD[AppTags] = rddMetadata.map(extractAppIdTags)

    // Unisce `appUserRDD` e `appTagsRDD` su `appId` per creare un RDD con (appId, tag, userId)
    val mergedRDD: RDD[(Int, String, String)] = recommendedRDD
      .join(tagsRDD) // Ottieni RDD[(appId, (userId, tag))]
      .map { case (appId, (userId, tag)) => (appId, tag, userId) } // Trasforma in (appId, tag, userId)
      .filter { case (_, tag, _) => tag.nonEmpty } // Filtra tag non vuoti
    //.persist(StorageLevel.MEMORY_AND_DISK)

    // Estrai `app_id` e `title` da `rddGames` come RDD[(appId, title)]
    val gameTitlesRDD: RDD[(Int, String)] = rddGames.map { line =>
      val fields = line.split(",")
      val appId = fields(0).toInt
      val title = fields(1)
      (appId, title)
    }

    // Raggruppa i titoli dei giochi per utente e accumula le parole
    val aggregatedTitlesByUser = mergedRDD
      .map { case (_, title, user) => (user, title.split("\\s+")) } // Suddivide il titolo in parole
      .reduceByKey(_ ++ _) // Accumula le parole per ogni utente

    // Esplode le parole aggregate in singoli record per ogni utente
    val explodedWordsByUser = aggregatedTitlesByUser
      .flatMap { case (userId, words) => words.map(word => (userId, word)) }

    (mergedRDD, explodedWordsByUser, gameTitlesRDD)
  }

  private def calculateTFIDF(explodeRdd: RDD[(String, String)]): RDD[(String, Map[String, Double])] = {

    // Funzione per calcolare TF
    def calculateTermFrequency(words: String): Map[String, Double] = {
      val wordsArray = words.split(",")
      val totalWords = wordsArray.length.toDouble
      wordsArray
        .groupBy(identity)
        .mapValues(_.length / totalWords)
    }

    // Funzione per calcolare IDF
    def calculateInverseDocumentFrequency(userWordsRDD: RDD[(String, String)]): Map[String, Double] = {

      val totalDocuments = userWordsRDD.count().toDouble

      userWordsRDD
        .flatMap { case (_, words) => words.split(",").distinct }
        .map(word => (word, 1))
        .reduceByKey(_ + _)
        .mapValues(count => math.log(totalDocuments / count))
        .collect()
        .toMap
    }

    // Funzione per calcolare TF-IDF
    def calculateTFIDFForUsers(userWordsRDD: RDD[(String, String)], tfFunction: String => Map[String, Double], idfFunction: RDD[(String, String)] => Map[String, Double]): RDD[(String, Map[String, Double])] = {

      val groupedWordsByUser = userWordsRDD
        .reduceByKey(_ + "," + _)
      // .persist(StorageLevel.MEMORY_AND_DISK)

      // Calcola IDF una volta sola per tutte le parole
      val idfValues = idfFunction(groupedWordsByUser)

      // Calcola TF-IDF per ciascun utente
      groupedWordsByUser.map { case (user, words) =>
        val tfValues = tfFunction(words)
        val tfidfValues = tfValues.map { case (word, tf) =>
          (word, tf * idfValues.getOrElse(word, 0.0))
        }
        (user, tfidfValues)
      }
    }

    // Calcola i valori TF-IDF finali usando le funzioni TF e IDF definite
    calculateTFIDFForUsers(explodeRdd, calculateTermFrequency, calculateInverseDocumentFrequency)
  }

  import spark.implicits

  private def getSimilarUsers[T: ClassTag](
                                            targetUser: Int,
                                            tfidfValues: RDD[(String, Map[String, Double])]
                                          ): Array[(String, Double)] = {

    // Funzione per calcolare la similarità coseno tra due vettori
    def computeCosineSimilarity(vector1: Map[String, Double], vector2: Map[String, Double]): Double = {

      // Calcolo del prodotto scalare tra i due vettori
      val dotProduct = vector1.foldLeft(0.0) { case (acc, (key, value)) =>
        acc + vector2.getOrElse(key, 0.0) * value
      }
      // Calcolo delle magnitudini di ciascun vettore
      val magnitude1 = math.sqrt(vector1.values.map(value => value * value).sum)
      val magnitude2 = math.sqrt(vector2.values.map(value => value * value).sum)
      // Calcolo della similarità coseno, con gestione della divisione per zero
      if (magnitude1 == 0 || magnitude2 == 0) 0.0 else dotProduct / (magnitude1 * magnitude2)
    }

    // Filtra il dataset per ottenere i giochi dell'utente target
    val targetUserGamesRDD = tfidfValues
      .filter { case (userId, _) => userId == targetUser.toString }
      .map { case (_, gameVector) => gameVector }

    // Step: verifica se l'utente target ha dei giochi
    if (!targetUserGamesRDD.isEmpty()) {
      // Colleziona i giochi dell'utente target
      val targetUserGames = targetUserGamesRDD.collect().head

      // Calcola la similarità per gli altri utenti
      val similarUsers = tfidfValues
        .filter(_._1 != targetUser.toString) // Esclude l'utente target
        .map { case (otherUserId, otherUserGames) =>
          (otherUserId, computeCosineSimilarity(targetUserGames, otherUserGames))
        }
        .collect() // Colleziona i risultati per il driver
        .sortBy(-_._2) // Ordina per similarità (in ordine decrescente)
        .take(3) // Prendi i top 3 utenti simili

      similarUsers // Ritorna la lista degli utenti simili
    } else {
      // Ritorna un array vuoto se l'utente target non ha giochi
      Array.empty[(String, Double)]
    }
  }

  private def getRecommendation(
                                 mergedRdd: RDD[(Int, String, String)],
                                 topUsersWithSimilarity: Array[(String, Double)],
                                 gamesData: RDD[(Int, String)],
                                 targetUser: Int
                               ): Unit = {

    // Ottieni gli appId giocati dall'utente target
    val appIdsPlayedByTargetUser = mergedRdd
      .filter { case (_, _, user) => user == targetUser.toString }
      .map(_._1)
      .distinct()
      .collect()
      .toSet

    // Ottieni i titoli dei giochi giocati dall'utente target
    val titlesPlayedByTargetUser = gamesData
      .filter { case (appId, _) => appIdsPlayedByTargetUser.contains(appId) }
      .map(_._2)
      .distinct()
      .collect()
      .toSet

    // Identifica i 3 utenti più simili
    val topSimilarUserIds = topUsersWithSimilarity.take(3).map(_._1).toSet

    println("Top 3 similar users:")
    topUsersWithSimilarity.take(3).foreach(println)

    // Filtra e mappa i giochi consigliati escludendo quelli già giocati dall'utente target
    val recommendedGames = mergedRdd
      .filter { case (_, gameTitle, user) =>
        topSimilarUserIds.contains(user) && !titlesPlayedByTargetUser.contains(gameTitle)
      }
      .map { case (appId, gameTitle, userId) => (appId, (gameTitle, userId)) }

    // Aggiungi i titoli dei giochi tramite join con gamesData
    val recommendationsWithTitle = recommendedGames
      .join(gamesData)
      .map { case (appId, ((_, userId), title)) => (appId, title, userId) }

    // Raggruppa le raccomandazioni per appId e aggrega gli userId
    val groupedRecommendations = recommendationsWithTitle
      .map { case (appId, title, userId) => (appId, (title, userId)) }
      .reduceByKey { case ((title, userId1), (_, userId2)) =>
        val userIds = Set(userId1, userId2).mkString(",") // Evita duplicati negli userId
        (title, userIds)
      }
      .map { case (appId, (title, userIds)) =>
        (appId, title, userIds)
      }

    // Stampa le raccomandazioni finali
    groupedRecommendations.collect().foreach { case (_, title, userIds) =>
      println(s"userId: $userIds, title: $title")
    }
  }
}
