package com.unibo.recommendationsystem

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession.builder

object recommendationRDD {
  def main(args: Array[String]): Unit = {

    //Initialize SparkSession
    val spark = builder
      .appName("recommendationsystem")
      .config("spark.master", "local[*]")
      .getOrCreate()

    val dataPathRec = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/recommendations.csv"
    val dataPathGames = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/games.csv"

    val t4 = System.nanoTime()

    //Load dataset as Dataframe
    val dfRec = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathRec)
    val dfGames = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathGames)


    val rddRec = dfRec.rdd
    val rddGames = dfGames.rdd


    //Select useful data from recommendations dataset
    val selectedRecRDD = rddRec.map { row =>
      val appId = row.getInt(0)
      val recommended = row.getBoolean(4)
      val user = row.getInt(6).toString
      (appId, recommended, user)
    }

    //Select useful data from games dataset
    val selectedGamesRDD = rddGames.map { row =>
      val appId = row.getInt(0)
      val title = row.getString(1)
      (appId, title)
    }

    //Creates a dictionary (Map) from the selectedGamesRDD. The appId becomes the key for fast lookups of game titles.
    val titleDict = selectedGamesRDD.collectAsMap()

    //Enriches the recommendation data (selectedRecRDD) by adding the title of each game.
    val mergedRDD = selectedRecRDD.map {
      case  (appId, recommended , user) =>
        val title = titleDict.getOrElse(appId, "") // Handle potential missing titles
        (appId, recommended, user, title)
    }

    def transformTitle(row: (Int, Boolean, String, String)): (Int, Boolean, String, String) = {
      val appId = row._1
      val recommended = row._2
      val user = row._3
      var title = row._4

      // Apply transformations to the title
      title = title.toLowerCase()
        .trim()
        .replaceAll("\\s+", " ")

      (appId, recommended, user, title)
    }

    // Clean the dataset from useless whitespaces
    val cleanMergeRDD = mergedRDD.map(transformTitle)

    val t5 = System.nanoTime()


    val t2 = System.nanoTime()


    def splitTitle(row: (Int, Boolean, String, String)): (Int, Boolean, String, Array[String]) = {
      val appId = row._1
      val recommended = row._2
      val user = row._3
      val title = row._4

      val words = title.split("\\s+")
      (appId, recommended, user, words)
    }

    //Tokenization of titles on whitespaces
    val datasetRDD = cleanMergeRDD.map(splitTitle)

    // Flatten the resulting sequence of array of words into a single array
    def flattenWordsRDD(wordLists: Seq[Array[String]]): Array[String] = {
      wordLists.flatten.toArray
    }

    // Perform the aggregation
    val aggregateDataRDD = datasetRDD
      .map { case (_, _, user, words) => (user, words) } // Map to (user, words)
      .groupByKey() // Group by user
      .mapValues { wordsIterable =>
        flattenWordsRDD(wordsIterable.toSeq) // Flatten words for each user
      }

    // println("aggregateDataRDD: " + aggregateDataRDD.count()) //aggregateDataRDD: 13781059
    // println("filteredDataRDD: " + filteredDataRDD.count()) //filteredDataRDD:     1178775

    // Filtering out all users with less than 20 words
    val filteredDataRDD = aggregateDataRDD.filter { case (_, words) => words.length >= 20 }

    // Explode the data
    val explodedRDD = aggregateDataRDD.flatMap { case (userId, words) =>
      words.map(word => (userId, word)) // Creates a new tuple (userId, word) for each individual word
    }

    // TF-IDF function definition
    def calculateTFIDF(userWordsDataset: RDD[(String, String)]): RDD[(String, Map[String, Double])] = {
      // Function to calculate term frequency (TF)
      def calculateTF(userWords: String): Map[String, Double] = {
        val wordsSplit = userWords.split(",")
        val totalWords = wordsSplit.size.toDouble
        val finalTF = wordsSplit.groupBy(identity).mapValues(_.length / totalWords)
        finalTF
      }

      // Function to calculate inverse document frequency (IDF)
      def calculateIDF(userWords: RDD[(String, String)]): Map[String, Double] = {
        val userCount = userWords.count()
        val wordsCount = userWords
          .flatMap { case (user, words) =>
            words.split(",").distinct.map(word => (user, word)) // Include docId and make words distinct
          }
          .map { case (user, word) => (word, user) } // Swap for grouping by word
          .groupByKey()
          .mapValues(_.toSet.size)

        val idfValues = wordsCount.map { case (word, count) => (word, math.log(userCount.toDouble / count)) }.collect().toMap[String,Double]
        idfValues
      }


      // Concatenates the words associated with each user into a comma-separated string
      val groupedUserWords = userWordsDataset
        .groupByKey() // Group by the userId
        .map { case (userId, wordsIterable) =>
          val wordsString = wordsIterable.mkString(",")
          (userId, wordsString)
        }

      val idfValues = calculateIDF(groupedUserWords)

      val tfidfValues = groupedUserWords.map { case (user, words) =>
        val termFrequencies = calculateTF(words)
        val tfidf = termFrequencies.map { case (word, tf) => (word, tf * idfValues.getOrElse(word, 0.0)) }
        (user, tfidf)
      }
      tfidfValues
    }


    val tfidfValues = calculateTFIDF(explodedRDD)

    val t3 = System.nanoTime()

    // Input: two vectors as a map of words and weights
    // Output: cosine similarity
    def computeCosineSimilarity(vector1: Map[String, Double], vector2: Map[String, Double]): Double = {
      def dotProduct(v1: Map[String, Double], v2: Map[String, Double]): Double = {
        v1.foldLeft(0.0) { case (acc, (key, value)) =>
          v2.get(key).map(value * _).getOrElse(0.0) + acc // Handle potential missing keys and type errors
        }
      }

      // Calculate vector magnitude (length)
      def magnitude(vector: Map[String, Double]): Double = {
        math.sqrt(vector.values.map(value => value * value).sum)
      }

      // Calculate cosine similarity
      dotProduct(vector1, vector2) / (magnitude(vector1) * magnitude(vector2))
    }

    val targetUser = 2591067

    val t0 = System.nanoTime()

    // Get users similar to the target
    def getSimilarUsers(userId: Int, tfidfValues: RDD[(String, Map[String, Double])]): Array[(String, Double)] = {

      val userGames = tfidfValues.filter(_._1.equals(userId.toString)).first()._2

      // Exclude the target user from recommendations
      tfidfValues.filter(_._1 != userId.toString) // Exclude the target user
        .map { case (otherUserId, otherUserGames) =>
          // Calculate similarity to given user
          (otherUserId, computeCosineSimilarity(userGames,otherUserGames)) // Calculate similarity here
        }
        .collect()
        .sortBy(-_._2) // Sort by highest score
        .take(10) // Take the three best matches
    }

    // Get recommendations for target users, based on previously calculated TF-IDF values
    val recommendations = getSimilarUsers(targetUser, tfidfValues)
    recommendations.foreach(println)

    /*
(10941911,0.7293625797795579)
(14044364,0.7263267622929318)
(4509885,0.7186991307198306)
(3278010,0.7159065615500113)
(6019065,0.7126999191199811)
(7889674,0.7113882151776377)
(1216253,0.7088144049757779)
(144350,0.7063527142603677)
(6222146,0.7033717175918999)
(10974221,0.7028838351034404)
     */

    // Extract games recommended by the target user
    val titlesPlayedByTargetUserRDD = cleanMergeRDD
      .filter { case (_, _, user, _) => user.equals(targetUser.toString)}
      .map { case (_, _, _, title) => title }
      .distinct()

    val titlesPlayedByTargetUserSet = titlesPlayedByTargetUserRDD.collect().toSet

    // Extract relevant user IDs from recommendations
    val userIdsToFind = recommendations.take(3).map(_._1).toSet

    // Filter datasetDF to remove already played games
    val filteredRDD = cleanMergeRDD.filter { case (_, isRecommended, user, title) =>
      userIdsToFind.contains(user) &&
        !titlesPlayedByTargetUserSet.contains(title) &&
        isRecommended
    }

    val finalRecommendations = filteredRDD.map { case (appId, _, user, title) =>
      ((appId, title), user)
    }

    val t1 = System.nanoTime()

    finalRecommendations.take(20).foreach(println)


    // Calculating execution times
    println("\n\nExecution time(recommendation):\t"+ (t1-t0)/1000000 + "ms\n")
    println("\n\nExecution time(Tf-Idf calculation):\t"+ (t3-t2)/1000000 + "ms\n")
    println("\n\nExecution time(preprocessing):\t"+ (t5-t4)/1000000 + "ms\n")

  }
}

