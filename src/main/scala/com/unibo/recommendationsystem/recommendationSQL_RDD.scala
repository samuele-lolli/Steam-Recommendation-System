package com.unibo.recommendationsystem

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, collect_list, concat_ws, count, countDistinct, explode, lower, map_from_arrays, regexp_replace, size, split, trim, udf}

object recommendationSQL_RDD {
  def main(args: Array[String]): Unit = {

    //Initialize SparkSession
    val spark = SparkSession
      .builder
      .appName("recommendationsystem")
      .config("spark.master", "local[*]")
      .getOrCreate()

    val dataPathRec = "C:\\Users\\samue\\recommendationsystem\\steam-dataset\\recommendations.csv"
    val dataPathGames = "C:\\Users\\samue\\recommendationsystem\\steam-dataset\\games.csv"

    val tPreProcessingI = System.nanoTime()

    // Load dataset as DataFrame
    val dfRec = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathRec)
    val dfGames = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathGames)

    // Select useful columns
    val selectedRec = dfRec.select("app_id", "user_id", "is_recommended")
    val selectedGames = dfGames.select("app_id", "title")

    // Merge the DataFrame for app_id
    val merged = selectedRec.join(selectedGames, Seq("app_id"), "inner")

    // Clean the dataset from useless whitespace
    val cleanMerge = merged.withColumn("title", lower(trim(regexp_replace(col("title"), "\\s+", " "))))

    // Tokenization of titles on whitespaces
    val dataset = cleanMerge.withColumn("words", split(col("title"), "\\s+"))

    // Converts nested sequences into a single list of strings, combining all inner lists
    val flattenWords: UserDefinedFunction = udf((s: Seq[Seq[String]]) => s.flatten)

    // Aggregate tokenized data by user ID
    val aggregateData = dataset.groupBy("user_id").agg(flattenWords(collect_list("words")).as("words"))

    // Filtering out all users with less than 20 words in their aggregated words list
    val filteredData = aggregateData.filter(size(col("words")) >= 20)

    // Explode the aggregated data for TF-IDF calculation
    val explodedDF = filteredData.withColumn("word", explode(col("words"))).select("user_id", "word")

    val tPreProcessingF = System.nanoTime()

    val tTFIDFI = System.nanoTime()

    // Calculate the total number of words associated with each unique user in the dataset
    val wordsPerUser = explodedDF.groupBy("user_id").agg(count("*").alias("total_words"))

    //Calculate the Term Frequency (TF) for each word and user combination
    val tf = explodedDF.groupBy("user_id", "word")
      .count()
      .withColumnRenamed("count", "term_count") // Rename to avoid ambiguity
      .join(wordsPerUser, "user_id")
      .withColumn("term_frequency", col("term_count") / col("total_words"))

    //Calculate Document Frequency
    val dfDF = explodedDF.groupBy("word")
      .agg(countDistinct("user_id").alias("document_frequency"))

    //Counting the total number of users.
    val totalDocs = filteredData.select(count("user_id")).first()
    //213364

    val rdd = dfDF.rdd

    //Calculate the IDF for each word in your RDD
    //IDF helps down-weight common words and emphasize terms that are more informative and discriminative
    val idfRDD = rdd.map { row =>
      val word = row.getString(0) // Assuming word is at index 0
      val docFreq = row.getLong(1) // Assuming document_frequency is at index 1
      val idf = math.log(totalDocs.toDouble / docFreq)
      (word, idf)
    }

    import spark.implicits._ // For toDF() method

    val idfDF = idfRDD.toDF("word", "idf")

    // Join the DataFrames on the 'word' column
    val tfidfDF = tf.join(idfDF, "word")
      .withColumn("tf_idf", col("term_frequency") * col("idf"))
      .select("user_id", "word", "tf_idf")

    val aggregatedXUser = tfidfDF.groupBy("user_id")
      .agg(concat_ws(",", collect_list("word")).alias("words"),
        concat_ws(",", collect_list(col("tf_idf").cast("string"))).alias("tf_idf_values"))

    val preprocessedDF = aggregatedXUser
      .withColumn("word_array", split(col("words"), ","))
      .withColumn("tfidf_array", split(col("tf_idf_values"), ","))
      .withColumn("tfidf_array", col("tfidf_array").cast("array<double>")) // Cast to double array
      .withColumn("word_tfidf_map", map_from_arrays(col("word_array"), col("tfidf_array")))

    val tTFIDFF = System.nanoTime()


    val tCosineSimilarityI = System.nanoTime()

    //Convert in a RDD[(Int, Map[String, Double])]
    val userSimRDD = preprocessedDF.rdd.map(row => {
      val userId = row.getAs[Int]("user_id")
      val wordTfidfMap = row.getAs[Map[String, Double]]("word_tfidf_map")
      (userId, wordTfidfMap)
    })

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

    // Get users similar to the target
    def getSimilarUsers(userId: Int, tfidfValues: RDD[(Int, Map[String, Double])]): Array[(Int, Double)] = {

      val userGames = tfidfValues.filter(_._1 == userId).first()._2

      // Exclude the target user from recommendations
      tfidfValues.filter(_._1 != userId) // Exclude the target user
        .map { case (otherUserId, otherUserGames) =>
          // Calculate similarity to given user
          (otherUserId, computeCosineSimilarity(userGames, otherUserGames)) // Calculate similarity here
        }.sortBy(-_._2) // Sort by highest score
        .collect()
        .take(15) // Take the three best matches
    }

    // Get recommendations for target users, based on previously calculated TF-IDF values
    val recommendedUsers = getSimilarUsers(targetUser, userSimRDD)

    println("recommendedUsers Top15")
    recommendedUsers.foreach(println)
    //println("Recommendations Top3")
    /*
      (6019065,0.7146400338963895)
      (8605254,0.6975084350476757)
      (6222146,0.6917861806899793)
     */

    val tCosineSimilarityF = System.nanoTime()

    val tFinalRecommendI = System.nanoTime()

    // Extract games recommended by the target user
    val titlesPlayedByTargetUser = cleanMerge
      .filter(col("user_id") === targetUser)
      .select("title")
      .distinct() // In case the target user has duplicates
      .as[String] // Convert DataFrame to Dataset[String]
      .collect()

    // Extract relevant user IDs from recommendations
    val userIdsToFind = recommendedUsers.map(_._1).toSet

    // Filter datasetDF to remove already played games
    val filteredDF = cleanMerge.filter(
      col("user_id").isin(userIdsToFind.toSeq: _*) && // User ID is recommended
        !col("title").isin(titlesPlayedByTargetUser: _*) &&
        col("is_recommended") === true
    )

    val finalRecommendations = filteredDF.toDF().drop(col("is_recommended"))
      .groupBy("app_id", "title")
      .agg(collect_list("user_id").alias("users"))

    val tFinalRecommendF = System.nanoTime()

    finalRecommendations.show()

    // Calculating execution times
    println("\n\nExecution time(preprocessing):\t"+ (tPreProcessingF-tPreProcessingI)/1000000 + "ms\n")
    println("\n\nExecution time(Tf-Idf calculation):\t"+ (tTFIDFF-tTFIDFI)/1000000 + "ms\n")
    println("\n\nExecution time(Cosine similarity calculation):\t"+ (tCosineSimilarityF-tCosineSimilarityI)/1000000 + "ms\n")
    println("\n\nExecution time(final recommendation):\t"+ (tFinalRecommendF-tFinalRecommendI)/1000000 + "ms\n")
    println("\n\nExecution time(total):\t"+ (tFinalRecommendF-tPreProcessingI)/1000000 + "ms\n")

    spark.stop()
  }
}