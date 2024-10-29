package com.unibo.recommendationsystem


import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Encoder, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.collection.Map

object tagSQL {
  def main(args: Array[String]): Unit = {
    // Initialize SparkSession
    val spark = SparkSession.builder
      .appName("recommendationsystem")
      // Executor Memory and Cores: Balanced to fit 2 executors per worker node
      .config("spark.executor.memory", "40g")          // Allocate about 80% memory for each executor
      .config("spark.executor.cores", "4")             // 4 cores per executor
      .config("spark.executor.instances", "4")         // Static executors for 2 primary workers (2 per worker)

      // Dynamic Allocation for Scaling with Spot Node
      .config("spark.dynamicAllocation.enabled", "true")
      .config("spark.dynamicAllocation.minExecutors", "4")   // Minimum executors: for 2 primary workers
      .config("spark.dynamicAllocation.maxExecutors", "5")   // Max executors: to include the preemptible node
      .config("spark.dynamicAllocation.executorIdleTimeout", "60s")

      // Speculative Execution to handle stragglers
      .config("spark.speculation", "true")
      .config("spark.speculation.interval", "100ms")
      .config("spark.speculation.multiplier", "1.5")

      // Shuffle and Serialization Optimizations
      .config("spark.sql.shuffle.partitions", "96")  // (4 cores * 4 executors * 6) for balanced parallelism
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.rdd.compress", "true")
      .config("spark.shuffle.file.buffer", "1m")     // Larger buffer for shuffling to avoid frequent disk I/O

      // Memory Management Adjustments
      .config("spark.memory.fraction", "0.7")        // Increase execution memory for large operations
      .config("spark.memory.storageFraction", "0.3") // Leave less for storage due to MEMORY_AND_DISK persistence

      // Adaptive Query Execution for optimized shuffling and partitioning
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.adaptive.coalescePartitions.enabled", "true") // Combine small partitions to reduce overhead

      .getOrCreate()



    val dataPathRec = "gs://dataproc-staging-us-central1-534461255477-conaqzw0/data/recommendations.csv"
    val dataPathGames = "gs://dataproc-staging-us-central1-534461255477-conaqzw0/data/games.csv"
    val metadataPath = "gs://dataproc-staging-us-central1-534461255477-conaqzw0/data/games_metadata.json"

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

    val dfRec = spark.read.format("csv").option("header", "true").schema(recSchema).load(dataPathRec).filter("is_recommended = true")//.sample(withReplacement = false, fraction = 0.25)
    val dfGames = spark.read.format("csv").option("header", "true").schema(gamesSchema).load(dataPathGames)
    val dfMetadata = spark.read.format("json").schema(metadataSchema).load(metadataPath)

    val selectedRec = dfRec.select("app_id", "user_id")
    val selectedGames = dfGames.select("app_id", "title")

    val merged = selectedRec.join(selectedGames, Seq("app_id"))
      .join(dfMetadata.drop("description"), Seq("app_id"))
      .filter(size(col("tags")) > 0)


    val cleanMerge = merged
      .withColumn("tags", transform(col("tags"), tag => lower(trim(regexp_replace(tag, "\\s+", " ")))))
      .withColumn("tagsString", concat_ws(",", col("tags")))  // Join tags with commas
      .drop("tags")
      .persist(StorageLevel.MEMORY_AND_DISK)


    // Tokenize by splitting on commas to maintain multi-word tags as single elements
    val filteredData = cleanMerge
      .withColumn("words", split(col("tagsString"), ","))  // Split on commas, preserving multi-word tags
      .groupBy("user_id")
      .agg(flatten(collect_list("words")).as("words"))


    // Explode aggregated data for TF-IDF calculation
    val explodedDF = filteredData.withColumn("word", explode(col("words"))).select("user_id", "word")
      .persist(StorageLevel.MEMORY_AND_DISK)


    val tPreProcessingF = System.nanoTime()

    val tTFIDFI = System.nanoTime()

    // Calculate total number of words for each user
    val wordsPerUser = explodedDF.groupBy("user_id").agg(count("*").alias("total_words"))

    // Calculate Term Frequency (TF)
    val tf = explodedDF.groupBy("user_id", "word")
      .agg(count("*").alias("term_count"))
      .join(wordsPerUser, "user_id")
      .withColumn("term_frequency", col("term_count") / col("total_words"))

    // Calculate Document Frequency (DF)
    val dfDF = explodedDF.groupBy("word")
      .agg(countDistinct("user_id").alias("document_frequency"))

    // Calculate total number of users
    val totalDocs = filteredData.count()

    // Calculate IDF for each word using SQL
    dfDF.createOrReplaceTempView("dfDF")
    val idfDF = spark.sql(s"""SELECT word, log($totalDocs / document_frequency) AS idf FROM dfDF""")

    // Join TF and IDF to get TF-IDF
    val tfidfDF = tf.join(idfDF, "word")
      .withColumn("tf_idf", col("term_frequency") * col("idf"))
      .select("user_id", "word", "tf_idf")

    val tTFIDFF = System.nanoTime()

    val targetUser = 2591067

    val tCosineSimilarityI = System.nanoTime()

    // Define the cosine similarity function
    def computeCosineSimilarity(vector1: Map[String, Double], vector2: Map[String, Double], dotProductFunc: (Map[String, Double], Map[String, Double]) => Double, magnitudeFunc: Map[String, Double] => Double): Double = {
      val magnitude1 = magnitudeFunc(vector1)
      val magnitude2 = magnitudeFunc(vector2)
      if (magnitude1 == 0.0 || magnitude2 == 0.0) 0.0 // Avoid division by zero
      else dotProductFunc(vector1, vector2) / (magnitude1 * magnitude2)
    }

    // Define the dot product and magnitude functions
    val dotProduct = (v1: Map[String, Double], v2: Map[String, Double]) => {
      v1.foldLeft(0.0) { case (acc, (key, value)) =>
        acc + v2.getOrElse(key, 0.0) * value
      }
    }

    val magnitude = (vector: Map[String, Double]) => {
      math.sqrt(vector.values.map(value => value * value).sum)
    }

    // Convert DataFrame rows to TF-IDF maps
    def rowToTfIdfMap(row: Row): Map[String, Double] = {
      row.getAs[Seq[Row]]("tags").map(tag => tag.getString(0) -> tag.getDouble(1)).toMap
    }

    // Step 1: Extract TF-IDF vector for the target user
    val targetUserData = tfidfDF.filter(col("user_id") === targetUser)
      .groupBy("user_id")
      .agg(collect_list(struct("word", "tf_idf")).alias("tags"))
    val targetUserVector = targetUserData.collect().headOption.map(rowToTfIdfMap).getOrElse(Map.empty[String, Double])

    // Step 2: Calculate cosine similarity with other users
    val otherUsersData = tfidfDF.filter(col("user_id") =!= targetUser)
      .groupBy("user_id")
      .agg(collect_list(struct("word", "tf_idf")).alias("tags"))

    import org.apache.spark.sql.Encoders

    // Define an implicit encoder for (Int, Double)
    implicit val tupleEncoder: Encoder[(Int, Double)] = Encoders.product[(Int, Double)]
    // Now you can use DataFrame map with the encoder
    val otherUsersWithSimilarity = otherUsersData.map { row =>
      val userId = row.getAs[Int]("user_id")
      val userVector = rowToTfIdfMap(row)
      val cosineSimilarity = computeCosineSimilarity(targetUserVector, userVector, dotProduct, magnitude)
      (userId, cosineSimilarity)
    }.toDF("user_id", "cosine_similarity")


    // Step 3: Get the top 3 users with highest cosine similarity
    val top3Users = otherUsersWithSimilarity.orderBy(desc("cosine_similarity")).limit(3)
    top3Users.show()

    /*
    13498880,65064,7002264
     */

    val tCosineSimilarityF = System.nanoTime()

    val tFinalRecommendI = System.nanoTime()

    val topSimilarUsers = top3Users.select("user_id").collect().map(row => row.getAs[Int]("user_id")).toList

    // Step 2: Fetch the games played by the similar users
    val gamesByTopUsers = dfRec.filter(col("user_id").isin(topSimilarUsers: _*))  // Use : _* to expand the list
      .select("app_id", "user_id")

    // Step 3: Fetch the games played by the target user
    val gamesByTargetUser = dfRec.filter(col("user_id") === targetUser)
      .select("app_id")

    // Step 4: Exclude the games played by the target user from the games played by the similar users
    val recommendedGames = gamesByTopUsers.join(gamesByTargetUser, Seq("app_id"), "left_anti")

    // Step 5: Join with dfGames to get the titles of the recommended games
    val finalRecommendations = recommendedGames
      .join(dfGames.select("app_id", "title"), Seq("app_id"))
      .select("title", "user_id")

    // Show the resulting DataFrame with titles and users
    val groupedRecommendations = finalRecommendations
      .groupBy("title")
      .agg(collect_list("user_id").alias("user_ids")) // Aggregate user_ids for each title
      .select("title", "user_ids") // Select only the title and aggregated user_ids

    groupedRecommendations.show(false) // Display the result without truncating

    val tFinalRecommendF = System.nanoTime()

    // Calculate and display execution times
    println(s"\n\nExecution time (preprocessing): ${(tPreProcessingF - tPreProcessingI) / 1000000} ms")
    println(s"\n\nExecution time (TF-IDF calculation): ${(tTFIDFF - tTFIDFI) / 1000000} ms")
    println(s"\n\nExecution time (cosine similarity calculation): ${(tCosineSimilarityF - tCosineSimilarityI) / 1000000} ms")
    println(s"\n\nExecution time (final recommendation): ${(tFinalRecommendF - tFinalRecommendI) / 1000000} ms")
    println(s"\n\nExecution time (total): ${(tFinalRecommendF - tPreProcessingI) / 1000000} ms")


    //LOCALE
    /*
    +--------------------+-------+
|               title|user_id|
+--------------------+-------+
|         Garry's Mod|7002264|
|DRAGON BALL Z: KA...|7002264|
|Total War: WARHAMMER|7002264|
|Halo: The Master ...|7002264|
|           Evil West|7002264|
|                GTFO|7002264|
|  Grand Theft Auto V|7002264|
|         War Thunder|7002264|
|Red Dead Redempti...|7002264|
|    Wallpaper Engine|7002264|
|Sea of Thieves 20...|7002264|
|      Cyberpunk 2077|7002264|
|          Green Hell|7002264|
|Age of Empires IV...|7002264|
|               Hades|7002264|
|          Subnautica|7002264|
|       Apex Legends™|7002264|
|       Call of Duty®|7002264|
|Warhammer: Vermin...|7002264|
|STAR WARS™: The O...|7002264|
+--------------------+-------+
only showing top 20 rows



Execution time (preprocessing): 2115 ms


Execution time (TF-IDF calculation): 62662 ms


Execution time (cosine similarity calculation): 413180 ms


Execution time (final recommendation): 268536 ms


Execution time (total): 746494 ms
     */


    //CLUSTER-PRIMO
    /*
    +--------+------------------+
| user_id| cosine_similarity|
+--------+------------------+
|   65064|0.8795445076711828|
|13498880|0.8783365585420777|
| 7002264|0.8752521190765519|
+--------+------------------+

+--------------------------------------+----------+
|title                                 |user_ids  |
+--------------------------------------+----------+
|100% Orange Juice                     |[7002264] |
|200% Mixed Juice!                     |[7002264] |
|3D Custom Lady Maker                  |[65064]   |
|60 Seconds!                           |[13498880]|
|60 Seconds! Reatomized                |[13498880]|
|9 Monkeys of Shaolin                  |[65064]   |
|99 Spirits                            |[7002264] |
|A Plague Tale: Requiem                |[65064]   |
|A Wild Catgirl Appears!               |[7002264] |
|ACE COMBAT™ 7: SKIES UNKNOWN          |[7002264] |
|Adorable Witch                        |[7002264] |
|Adorable Witch 2                      |[7002264] |
|Adorable Witch 3                      |[65064]   |
|Age of Empires IV: Anniversary Edition|[7002264] |
|Aimlabs                               |[13498880]|
|Akin Vol 2                            |[13498880]|
|Alan Wake                             |[65064]   |
|Alien Shooter                         |[13498880]|
|Among Us                              |[7002264] |
|Apex Legends™                         |[7002264] |
+--------------------------------------+----------+
only showing top 20 rows



Execution time (preprocessing): 4770 ms


Execution time (TF-IDF calculation): 49883 ms


Execution time (cosine similarity calculation): 147905 ms


Execution time (final recommendation): 101625 ms


Execution time (total): 304185 ms
     */

    spark.stop()
  }
}
