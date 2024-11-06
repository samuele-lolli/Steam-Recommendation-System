package com.unibo.recommendationsystem

import com.unibo.recommendationsystem.utils.{schemaUtils, timeUtils}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.reflect.ClassTag

class rddRecommendation (spark: SparkSession, dataRec: Dataset[Row], dataGames: DataFrame, metadata: DataFrame) {

  // Logica di raccomandazione
  def recommend(targetUser: Int): Unit = {
    // Time the preprocessing of data
    println("Preprocessing data...")
    //val tPreProcessingI = System.nanoTime()
    val (mergedRdd, explodedRDD, gamesData) = timeUtils.time(preprocessData(), "Preprocessing Data", "RDD")
    //val (mergedRdd, explodedRDD, gamesData) = preprocessData()
    //val tPreProcessingF = System.nanoTime()

    // Time the TF-IDF calculation
    println("Calculate term frequency and inverse document frequency...")
    //val tTFIDFI = System.nanoTime()
    val tfidfValues = timeUtils.time(calculateTFIDF(explodedRDD), "Calculating TF-IDF", "RDD")
    //val tfidfValues = calculateTFIDF(explodedRDD)
    //val tTFIDFF = System.nanoTime()

    // Time the similarity computation
    println("Calculate cosine similarity to get similar users...")
    //val tCosineSimilarityI = System.nanoTime()
    val topUsersSimilarity = timeUtils.time(getSimilarUsers(targetUser, tfidfValues), "Getting Similar Users", "RDD")
    //val topUsersSimilarity = getSimilarUsers(targetUser, tfidfValues)
    //val tCosineSimilarityF = System.nanoTime()

    // Time the recommendation generation
    println("Calculate final recommendation...")
    //val tFinalRecommendI = System.nanoTime()
    timeUtils.time(getRecommendation(mergedRdd, topUsersSimilarity, gamesData, targetUser), "Generating Recommendations", "RDD")
   //getRecommendation(mergedRdd, topUsersSimilarity, gamesData, targetUser)
    //val tFinalRecommendF = System.nanoTime()

/*
    println(s"\n\nExecution time(preprocessing): ${(tPreProcessingF - tPreProcessingI) / 1000000}ms\n")
    println(s"\n\nExecution time(Tf-Idf calculation): ${(tTFIDFF - tTFIDFI) / 1000000}ms\n")
    println(s"\n\nExecution time(Cosine similarity calculation): ${(tCosineSimilarityF - tCosineSimilarityI) / 1000000}ms\n")
    println(s"\n\nExecution time(final recommendation): ${(tFinalRecommendF - tFinalRecommendI) / 1000000}ms\n")
    println(s"\n\nExecution time(total): ${(tFinalRecommendF - tPreProcessingI) / 1000000}ms\n")

 */

    /*
    Elapsed time for Generating Recommendations:	41423ms (41423732625ns)
     */
    //Total time of execution: 427211ms

  }

  private def preprocessData() : (RDD[(Int, String, String)], RDD[(String, String)], RDD[(Int, String)]) = {

    def processRDD[T: ClassTag](rdd: RDD[org.apache.spark.sql.Row], processingFunc: org.apache.spark.sql.Row => T): RDD[T] = {
      rdd.map(processingFunc)
    }

    // Change the structure to include userId
    val selectedRecRDD: RDD[(Int, String)] = processRDD(dataRec.rdd, row => (row.getInt(0), row.getInt(6).toString))

    // Step 2: Create a higher-order function to map game titles
    def mapGames[T](rdd: RDD[org.apache.spark.sql.Row], mappingFunc: org.apache.spark.sql.Row => (Int, T)): Map[Int, T] = {
      rdd.map(mappingFunc).collect().toMap
    }

    // Create title dictionary mapping appId to title
    val tags = mapGames(metadata.rdd, row => (row.getInt(0), row.getList(2).toArray.map(_.toString).mkString(",").toLowerCase.trim.replaceAll("\\s+", " ")))

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


    val gamesTitlesRDD = dataGames.rdd.map(row => (row.getAs[Int]("app_id"), row.getAs[String]("title")))



    (mergedRDD, explodedRDD, gamesTitlesRDD)
  }

  private def calculateTFIDF(explodedRdd: RDD[(String, String)]): RDD[(String, Map[String, Double])] = {

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

    val tfidfValues = calculateTFIDF(explodedRdd, calculateTF, calculateIDF)

    tfidfValues
  }

  private def getSimilarUsers[T: ClassTag](
                                            targetUser: Int,
                                            tfidfValues: RDD[(String, Map[String, Double])]
                                          ): Array[(String, Double)] = {

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

    def getSimilarUsers[T : ClassTag](
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

    val recommendations = getSimilarUsers(targetUser, tfidfValues, cosineSimilarity)

    recommendations
  }

  private def getRecommendation(
                                 mergedRdd: RDD[(Int, String, String)],
                                 topUsersXSimilarity: Array[(String, Double)],
                                 gamesData: RDD[(Int, String)],
                                 targetUser: Int
                               ): Unit = {

    val appIdsPlayedByTargetUser = mergedRdd
      .filter { case (_, _, user) => user == targetUser.toString }  // Filter by targetUser
      .map(_._1)  // Extract appId
      .distinct()  // Get unique appIds
      .collect()   // Collect appIds into an array
      .toSet       // Convert to a Set for easy lookup

    // Convert dfGames to an RDD of (appId, title)

    // Filter dfGamesRDD by appId and extract titles
    val titlesPlayedByTargetUser = gamesData
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

    val userIdsToFind = topUsersXSimilarity.take(3).map(_._1).toSet

    /*
    Top 3 similar users
(8971360,0.88100129281368)
(9911449,0.8785642563919683)
(11277999,0.8678767593227635)
     */

    val finalRecommendations = filterAndMap(mergedRdd,
      { case (_, tag, user) => userIdsToFind.contains(user) && !titlesPlayedByTargetUser.contains(tag)},
      { case (appId, tag, user) => (appId, tag, user) })

    // Step 2: Prepare finalRecommendations by mapping the appId to the title
    val finalRecommendationsWithTitle = finalRecommendations
      .map { case (appId, tags, userId) => (appId, (tags, userId)) }  // Prepare for join
      .join(gamesData)  // Join with the RDD containing (appId, title)
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
  }


}
/*
Execution time(preprocessing): 4669ms
Execution time(Tf-Idf calculation): 171363ms
Execution time(Cosine similarity calculation): 126298ms
Execution time(final recommendation): 7548ms
Execution time(total): 309878ms


Elapsed time for Preprocessing Data:	4164ms (4164205542ns)
Elapsed time for Calculating TF-IDF:	198220ms (198220372000ns)
Elapsed time for Getting Similar Users:	134354ms (134354478125ns)
Elapsed time for Generating Recommendations:	9320ms (9320373250ns)


Execution time(preprocessing): 4164ms
Execution time(Tf-Idf calculation): 198220ms
Execution time(Cosine similarity calculation): 134355ms
Execution time(final recommendation): 9323ms
Execution time(total): 346063ms

 */
