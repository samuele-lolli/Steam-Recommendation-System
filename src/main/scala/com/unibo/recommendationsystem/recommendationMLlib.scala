package com.unibo.recommendationsystem

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.UserDefinedFunction

object recommendationMLlib {

  def main(args: Array[String]): Unit = {
    // Initialize SparkSession
    val spark = SparkSession.builder
      .appName("recommendationsystem")
      .config("spark.master", "local[*]")
      .getOrCreate()

    // Paths for dataset
    val dataPathRec = "C:\\Users\\samue\\recommendationsystem\\steam-dataset\\recommendations.csv"
    val dataPathGames = "C:\\Users\\samue\\recommendationsystem\\steam-dataset\\games.csv"

    val tPreProcessingI = System.nanoTime()

    // Load datasets
    val dfRec = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathRec)
    val dfGames = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathGames)

    // Select relevant columns and join datasets on app_id
    val selectedRec = dfRec.select("app_id", "user_id", "is_recommended")
    val selectedGames = dfGames.select("app_id", "title")
    val merged = selectedRec.join(selectedGames, Seq("app_id"))

    val cleanMerge = merged.withColumn("title", lower(trim(regexp_replace(col("title"), "\\s+", " "))))

    // UDF to flatten nested sequences into a single list
    val flattenWords: UserDefinedFunction = udf((s: Seq[Seq[String]]) => s.flatten)

    // Tokenize the titles
    val tokenizer = new Tokenizer().setInputCol("title").setOutputCol("words")
    val tokenizedData = tokenizer.transform(cleanMerge)

    // Aggregate tokenized data by user ID and filter out users with less than 20 words
    val aggregateData = tokenizedData.groupBy("user_id").agg(flattenWords(collect_list("words")).as("words"))
    val filteredData = aggregateData.filter(size(col("words")) >= 20)

    val tPreProcessingF = System.nanoTime()

    val tTFIDFI = System.nanoTime()

    import spark.implicits._
    // Convert words to feature vectors using HashingTF and IDF
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("hashedFeatures").setNumFeatures(20000)
    val featurizedData = hashingTF.transform(filteredData)
    val idf = new IDF().setInputCol("hashedFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)

    val tTFIDFF = System.nanoTime()

    val tCosineSimilarityI = System.nanoTime()

    // UDF to convert sparse vectors to dense vectors for cosine similarity
    val denseVector = udf { (v: Vector) => Vectors.dense(v.toArray) }
    val dfWithDenseFeatures = rescaledData.withColumn("dense_features", denseVector(col("features")))

    // Target user ID
    val targetUser = 2591067
    val targetUserFeatures = dfWithDenseFeatures.filter($"user_id" === targetUser)
      .select("dense_features").first().getAs[Vector]("dense_features")

    // Compute cosine similarity
    val cosSimilarity = udf { (denseFeatures: Vector) =>
      val dotProduct = targetUserFeatures.dot(denseFeatures)
      val normA = Vectors.norm(targetUserFeatures, 2.0)
      val normB = Vectors.norm(denseFeatures, 2.0)
      dotProduct / (normA * normB)
    }

    // Compute and display cosine similarity for all users
    val usersSimilarity = dfWithDenseFeatures.filter($"user_id" =!= targetUser)
      .withColumn("cosine_sim", cosSimilarity(col("dense_features")))
      .select("user_id", "cosine_sim")
      .orderBy($"cosine_sim".desc)
      .limit(3)

    println("Top 3 similar users:")
    usersSimilarity.show()

    /*
    +--------+------------------+
    | user_id|        cosine_sim|
    +--------+------------------+
    |10941911|0.7374018640927337|
    |14044364|0.7310713056820088|
    | 7889674|0.7267982562572204|
    +--------+------------------+
    */
    val tCosineSimilarityF = System.nanoTime()
    val tFinalRecommendI = System.nanoTime()

    // Extract games recommended by the target user
    val titlesPlayedByTargetUser = cleanMerge.filter($"user_id" === targetUser)
      .select("title").distinct().as[String].collect()

    // Extract relevant user IDs from recommendations
    val userIdsToFind = usersSimilarity.select("user_id").as[Int].collect.toSet

    // Filter dataset to remove already played games and aggregate recommendations
    val finalRecommendations = cleanMerge.filter(col("user_id").isin(userIdsToFind.toArray: _*)
        && !col("title").isin(titlesPlayedByTargetUser: _*)
        && col("is_recommended") === true)
      .groupBy("app_id", "title")
      .agg(collect_list("user_id").alias("users"))

    val tFinalRecommendF = System.nanoTime()

    finalRecommendations.take(100).foreach(println)

    /*
    [1085660,destiny 2,WrappedArray(14044364)]
    [1172470,apex legends™,WrappedArray(14044364)]
    [1263370,seek girl:fog ⅰ,WrappedArray(7889674)]
    [307690,sleeping dogs: definitive edition,WrappedArray(14044364)]
    [1267910,melvor idle,WrappedArray(14044364)]
    [1126290,lost,WrappedArray(7889674, 14044364)]
    [1205240,tentacle girl,WrappedArray(7889674)]
    [1060670,taboos: cracks,WrappedArray(7889674)]
    [1509090,seek girl ⅷ,WrappedArray(7889674)]
    [1146630,yokai's secret,WrappedArray(7889674)]
    */

    // Calculate and display execution times
    println(s"\n\nExecution time (preprocessing): ${(tPreProcessingF - tPreProcessingI) / 1000000} ms")
    println(s"\n\nExecution time (Tf-Idf calculation): ${(tTFIDFF - tTFIDFI) / 1000000} ms")
    println(s"\n\nExecution time (cosine similarity calculation): ${(tCosineSimilarityF - tCosineSimilarityI) / 1000000} ms")
    println(s"\n\nExecution time (final recommendation): ${(tFinalRecommendF - tFinalRecommendI) / 1000000} ms")
    println(s"\n\nExecution time (total): ${(tFinalRecommendF - tPreProcessingI) / 1000000} ms")

    spark.stop()
  }
}
