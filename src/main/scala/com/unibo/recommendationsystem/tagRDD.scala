package com.unibo.recommendationsystem

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession.builder
import org.apache.spark.sql.types.{ArrayType, BooleanType, DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.storage.StorageLevel

import scala.reflect.ClassTag

object tagRDD {
  def main(args: Array[String]): Unit = {

    //Initialize SparkSession
    val spark = builder
      .appName("recommendationsystem")
       .config("spark.master", "local[*]")
      /* .config("spark.executor.memory", "48g") // Allocate 48 GB for each executor
        .config("spark.driver.memory", "8g")    // Allocate 8 GB for the driver
        .config("spark.executor.cores", "4")    // Use 4 cores per executor for parallelism
        .config("spark.default.parallelism", "32") // Set parallelism for transformations
        .config("spark.sql.shuffle.partitions", "32") // Optimize shuffle partitions
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.minExecutors", "2")
        .config("spark.dynamicAllocation.maxExecutors", "6")
       */
      .getOrCreate


    val dataPathRec = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/recommendations.csv"
    val dataPathGames = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/games.csv"
    val metadataPath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/games_metadata.json"

    val recSchema = StructType(Array(
      StructField("app_id", IntegerType, nullable = false), // ID del gioco
      StructField("helpful", IntegerType, nullable = true), // Numero di voti utili
      StructField("funny", IntegerType, nullable = true), // Numero di voti divertenti
      StructField("date", StringType, nullable = true), // Data della recensione
      StructField("is_recommended", BooleanType, nullable = true), // Recensione positiva o negativa
      StructField("hours", DoubleType, nullable = true), // Ore di gioco
      StructField("user_id", IntegerType, nullable = false), // ID utente
      StructField("review_id", IntegerType, nullable = false) // ID recensione
    ))

    val gamesSchema = StructType(Array(
      StructField("app_id", IntegerType, nullable = false), // ID del gioco
      StructField("title", StringType, nullable = true), // Titolo del gioco
      StructField("date_release", StringType, nullable = true), // Data di rilascio
      StructField("win", BooleanType, nullable = true), // Disponibile per Windows
      StructField("mac", BooleanType, nullable = true), // Disponibile per Mac
      StructField("linux", BooleanType, nullable = true), // Disponibile per Linux
      StructField("rating", StringType, nullable = true), // Valutazione del gioco
      StructField("positive_ratio", IntegerType, nullable = true), // Percentuale di recensioni positive
      StructField("user_reviews", IntegerType, nullable = true), // Numero di recensioni utente
      StructField("price_final", DoubleType, nullable = true), // Prezzo finale
      StructField("price_original", DoubleType, nullable = true), // Prezzo originale
      StructField("discount", DoubleType, nullable = true), // Sconto
      StructField("steam_deck", BooleanType, nullable = true) // CompatibilitÃ con Steam Deck
    ))

    val metadataSchema = StructType(Array(
      StructField("app_id", IntegerType, nullable = false),
      StructField("description", StringType, nullable = true),
      StructField("tags", ArrayType(StringType), nullable = true) // Array di stringhe per i tag
    ))

    //PREPROCESSING
    val tPreProcessingI = System.nanoTime()

    val dfRec = spark.read.format("csv").option("header", "true").schema(recSchema).load(dataPathRec).filter("is_recommended = true")//.sample(withReplacement = false, fraction = 0.35)
    val dfGames = spark.read.format("csv").option("header", "true").schema(gamesSchema).load(dataPathGames)
    val dfMetadata = spark.read.format("json").schema(metadataSchema).load(metadataPath)

    // Step 1: Use Higher-order function to process the recommendation RDD
    def processRDD[T: ClassTag](rdd: RDD[org.apache.spark.sql.Row], processingFunc: org.apache.spark.sql.Row => T): RDD[T] = {
      rdd.map(processingFunc)
    }

    // Change the structure to include userId
    val selectedRecRDD: RDD[(Int, String)] = processRDD(dfRec.rdd, row => (row.getInt(0), row.getInt(6).toString))

    // Step 2: Create a higher-order function to map game titles
    def mapGames[T](rdd: RDD[org.apache.spark.sql.Row], mappingFunc: org.apache.spark.sql.Row => (Int, T)): Map[Int, T] = {
      rdd.map(mappingFunc).collect().toMap
    }

    // Create title dictionary mapping appId to title
    val tags = mapGames(dfMetadata.rdd, row => (row.getInt(0), row.getList(2).toArray.map(_.toString).mkString(",").toLowerCase.trim.replaceAll("\\s+", " ")))

    // Broadcast the tagMap to all workers
    val broadcastTagMap = spark.sparkContext.broadcast(tags)

    // Update mergeRDDs to no longer require tagMap as a parameter
    def mergeRDDs[T: ClassTag](recRDD: RDD[(Int, String)], mapFunc: (Int, String) => T): RDD[T] = {
      recRDD.map { case (appId, userId) => mapFunc(appId, userId) }
    }

    // Update mapFunc to use broadcastTagMap instead of passing tagMap
    val mergedRDD = mergeRDDs(selectedRecRDD, (appId: Int, userId: String) => {
      val tag = broadcastTagMap.value.getOrElse(appId, "")  // Retrieve the game title
      (appId, tag, userId)  // Now include userId in the output
    })
      .filter { case (_, tag, _) => tag.nonEmpty }  // Filter out empty tags
      .persist(StorageLevel.MEMORY_AND_DISK)


    // Step 4: Aggregate data by user using a higher-order function to pass a filtering condition
    def aggregateByUser[T](rdd: RDD[(String, Array[String])], filterFunc: Array[String] => Boolean): RDD[(String, Array[String])] = {
      rdd.filter { case (_, words) => filterFunc(words) }
    }

    // Define filtering logic as a function
    val minWords = 0
    val filterCondition = (words: Array[String]) => words.length >= minWords

    val aggregateDataRDD = mergedRDD
      .map { case (_, title, user) => (user, title.split("\\s+")) } // Split title into words
      .reduceByKey { (arr1, arr2) => arr1 ++ arr2 } // Concatenate arrays of words

    val filteredAggregateDataRDD = aggregateByUser(aggregateDataRDD, filterCondition)

    def explodeRDD[T: ClassTag](rdd: RDD[(String, Array[String])], explodeFunc: (String, Array[String]) => Iterable[T]): RDD[T] = {
      rdd.flatMap { case (userId, words) => explodeFunc(userId, words) }
    }

    val explodedRDD = explodeRDD(filteredAggregateDataRDD, (userId, words) => words.map(word => (userId, word)))

    val tPreProcessingF = System.nanoTime()

    val tTFIDFI = System.nanoTime()

    // Step 6: TF-IDF function definition using higher-order functions
    def calculateTFIDF[T](userWordsDataset: RDD[(String, String)],
                          tfFunc: String => Map[String, Double],
                          idfFunc: RDD[(String, String)] => Map[String, Double]): RDD[(String, Map[String, Double])] = {

      // Group words by user
      val groupedUserWords = userWordsDataset
        .map { case (userId, words) => (userId, words) } // Keep as a (userId, words) pair
        .reduceByKey { (words1, words2) => words1 + "," + words2 } // Concatenate words with a comma
        .persist(StorageLevel.MEMORY_AND_DISK)

      // Use the provided IDF function
      val idfValues = idfFunc(groupedUserWords)

      // Calculate TF-IDF
      groupedUserWords.map { case (user, words) =>
        val tfValues = tfFunc(words)
        val tfidfValues = tfValues.map { case (word, tf) => (word, tf * idfValues.getOrElse(word, 0.0)) }
        (user, tfidfValues)
      }
    }

    // Define term frequency (TF) calculation logic
    val calculateTF = (userWords: String) => {
      val wordsSplit = userWords.split(",")
      val totalWords = wordsSplit.size.toDouble
      wordsSplit.groupBy(identity).mapValues(_.length / totalWords)
    }

    // Define inverse document frequency (IDF) calculation logic
    val calculateIDF = (userWords: RDD[(String, String)]) => {
      val userCount = userWords.count()  // Total number of users (or documents)

      // Directly compute IDF without storing intermediate 'wordsCount'
      userWords
        .flatMap { case (_, words) => words.split(",").distinct }  // Split and get distinct words
        .map(word => (word, 1))  // Map each distinct word to (word, 1)
        .reduceByKey(_ + _)      // Reduce by key to get the count of each word
        .map { case (word, count) => (word, math.log(userCount.toDouble / count)) }  // Compute IDF
        .collect()
        .toMap
    }

    val tfidfValues = calculateTFIDF(explodedRDD, calculateTF, calculateIDF)

    val tTFIDFF = System.nanoTime()

    val tCosineSimilarityI = System.nanoTime()


    // Step 7: Higher-order function to compute cosine similarity
    def computeCosineSimilarity(vector1: Map[String, Double], vector2: Map[String, Double], dotProductFunc: (Map[String, Double], Map[String, Double]) => Double, magnitudeFunc: Map[String, Double] => Double): Double = {
      dotProductFunc(vector1, vector2) / (magnitudeFunc(vector1) * magnitudeFunc(vector2))
    }

    // Define dot product and magnitude logic
    val dotProduct = (v1: Map[String, Double], v2: Map[String, Double]) => {
      v1.foldLeft(0.0) { case (acc, (key, value)) =>
        acc + v2.getOrElse(key, 0.0) * value
      }
    }

    val magnitude = (vector: Map[String, Double]) => {
      math.sqrt(vector.values.map(value => value * value).sum)
    }

    // Step 8: Higher-order function to get similar users

    def getSimilarUsers[T: ClassTag](
                                      targetUser: Int,
                                      tfidfValues: RDD[(String, Map[String, Double])],
                                      similarityFunc: (Map[String, Double], Map[String, Double]) => Double
                                    ): Array[(String, Double)] = {

      // Filter to get the target user's games

      val targetUserGamesRDD = tfidfValues
        .filter { case (userId, _) => userId == targetUser.toString }
        .map { case (_, gameVector) => gameVector }
      //println(s"Time for filtering: ${(tFilterEnd - tFilterStart) / 1e9} seconds")

      // Step 2: Check if the target user has any games
      if (!targetUserGamesRDD.isEmpty()) {
        // Step 3: Collect the target user's games (should be a single map)
        val targetUserGames = targetUserGamesRDD.collect().head
        // println(s"Time for collecting: ${(tCollectEnd - tCollectStart) / 1e9} seconds")


        // Step 3: Compute similarity for other users
        val similarUsers = tfidfValues
          .filter(_._1 != targetUser.toString) // Filter out the target user
          .map { case (otherUserId, otherUserGames) =>
            (otherUserId, similarityFunc(targetUserGames, otherUserGames))
          }
          .collect() // Collect the results to the driver
          .sortBy(-_._2) // Sort by similarity (descending)
          .take(3) // Take top 10 similar users

        similarUsers // Return the list of similar users
      } else {
        // Step 4: Return an empty array if the target user has no games
        Array.empty[(String, Double)]
      }
    }

    // Use cosine similarity logic
    val cosineSimilarity = (v1: Map[String, Double], v2: Map[String, Double]) =>
      computeCosineSimilarity(v1, v2, dotProduct, magnitude)

    val targetUser = 4893896
    val recommendations = getSimilarUsers(targetUser, tfidfValues, cosineSimilarity)

    val tCosineSimilarityF = System.nanoTime()

    // Get final recommendations
    val tFinalRecommendI = System.nanoTime()

    // Extract appIds played by target user
    val appIdsPlayedByTargetUser = mergedRDD
      .filter { case (_, _, user) => user == targetUser.toString }  // Filter by targetUser
      .map(_._1)  // Extract appId
      .distinct()  // Get unique appIds
      .collect()   // Collect appIds into an array
      .toSet       // Convert to a Set for easy lookup

    // Convert dfGames to an RDD of (appId, title)
    val dfGamesRDD = dfGames.rdd.map(row => (row.getAs[Int]("app_id"), row.getAs[String]("title")))

    // Filter dfGamesRDD by appId and extract titles
    val titlesPlayedByTargetUser = dfGamesRDD
      .filter { case (appId, _) => appIdsPlayedByTargetUser.contains(appId) }  // Filter by appId in the set
      .map(_._2)  // Extract the titles
      .distinct() // Ensure unique titles
      .collect()  // Collect the titles into an array
      .toSet      // Convert to a Set of titles


    // Use higher-order function for filtering and mapping final recommendations
    def filterAndMap[T: ClassTag](rdd: RDD[(Int, String, String)],
                                  filterFunc: ((Int, String, String)) => Boolean,
                                  mapFunc: ((Int, String, String)) => T): RDD[T] = {
      rdd.filter(filterFunc).map(mapFunc)
    }

    val userIdsToFind = recommendations.take(3).map(_._1).toSet

    println("Top 3 similar users")
    recommendations.take(3).foreach(println)
    /*
    Top 3 similar users
(8971360,0.88100129281368)
(9911449,0.8785642563919683)
(11277999,0.8678767593227635)
     */

    val finalRecommendations = filterAndMap(mergedRDD,
      { case (_, tag, user) => userIdsToFind.contains(user) && !titlesPlayedByTargetUser.contains(tag)},
      { case (appId, tag, user) => (appId, tag, user) })

    // Step 2: Prepare finalRecommendations by mapping the appId to the title
    val finalRecommendationsWithTitle = finalRecommendations
      .map { case (appId, tags, userId) => (appId, (tags, userId)) }  // Prepare for join
      .join(dfGamesRDD)  // Join with the RDD containing (appId, title)
      .map { case (appId, ((_, userId), title)) => (appId, title, userId) } // Replace tags with title

    val groupedRecommendations = finalRecommendationsWithTitle
      .map { case (appId, title, userId) => (appId, (title, userId)) }  // Prepare for grouping
      .reduceByKey { case ((title1, userId1), (_, userId2)) =>
        // Assuming title is the same for all records with the same appId
        val title = title1 // or title2, both are the same
        val userIds = Set(userId1, userId2).mkString(",") // Use a Set to avoid duplicates
        (title, userIds)  // This maintains userIds as a Set
      }
      .map { case (appId, (title, userIds)) =>
        (appId, title, userIds)  // Return as (appId, title, comma-separated userIds)
      }

    groupedRecommendations.collect().foreach { case (_, title, userIds) =>
      println(s"userId: $userIds, title: $title")
    }

    val tFinalRecommendF = System.nanoTime()


    println(s"\n\nExecution time(preprocessing): ${(tPreProcessingF - tPreProcessingI) / 1000000}ms\n")
    println(s"\n\nExecution time(Tf-Idf calculation): ${(tTFIDFF - tTFIDFI) / 1000000}ms\n")
    println(s"\n\nExecution time(Cosine similarity calculation): ${(tCosineSimilarityF - tCosineSimilarityI) / 1000000}ms\n")
    println(s"\n\nExecution time(final recommendation): ${(tFinalRecommendF - tFinalRecommendI) / 1000000}ms\n")
    println(s"\n\nExecution time(total): ${(tFinalRecommendF - tPreProcessingI) / 1000000}ms\n")

    spark.stop()
  }
}