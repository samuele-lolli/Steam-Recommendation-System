package com.unibo.recommendationsystem
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._


object recommendationMLlib {

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

    val dfRec = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathRec)
    val dfGames = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathGames)

    // Select useful columns
    val selectedRec = dfRec.select("app_id", "user_id", "is_recommended")
    val selectedGames = dfGames.select("app_id", "title")

    // Merge the DataFrame for app_id
    val merged = selectedRec.join(selectedGames, Seq("app_id"), "inner")

    // Clean the dataset from useless whitespace
    val cleanMerge = merged.withColumn("title", lower(trim(regexp_replace(col("title"), "\\s+", " "))))

    // Converts nested sequences into a single list of strings, combining all inner lists
    val flattenWords: UserDefinedFunction = udf((s: Seq[Seq[String]]) => s.flatten)

    val tokenizer = new Tokenizer()
      .setInputCol("title")
      .setOutputCol("words")
    val tokenizedData = tokenizer.transform(cleanMerge)

    // Aggregate tokenized data by user ID
    val aggregateData = tokenizedData.groupBy("user_id").agg(flattenWords(collect_list("words")).as("words"))

    // Filtering out all users with less than 20 words in their aggregated words list
    val filteredData = aggregateData.filter(size(col("words")) >= 20)

    val tPreProcessingF = System.nanoTime()

    val tTFIDFI = System.nanoTime()

    //HashingTF converts a collection of words into a fixed-length feature vector
    val hashingTF = new HashingTF()
      .setInputCol("words")
      .setOutputCol("hashedFeatures")
      .setNumFeatures(20000) // Adjust numFeatures as needed
    val featurizedData = hashingTF.transform(filteredData)

    //IDF helps reduce the impact of common terms like "the", "and" ...
    //Terms that are rare but appear in specific documents are often more informative. IDF boosts the importance of these terms
    val idf = new IDF().setInputCol("hashedFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)

    val tTFIDFF = System.nanoTime()

    val tCosineSimilarityI = System.nanoTime()

    import spark.implicits._
    // Extract dense features for cosine similarity
    val denseVector = udf { (v: Vector) => Vectors.dense(v.toArray) }
    val dfWithDenseFeatures = rescaledData.withColumn("dense_features", denseVector(col("features")))
    // Select target user
    val targetUser = 2591067
    val targetUserFeatures = dfWithDenseFeatures.filter($"user_id" === targetUser).select("user_id", "dense_features").first().getAs[Vector]("dense_features")

    // Compute cosine similarity
    val cosSimilarity = udf { (denseFeatures: Vector) =>
      val dotProduct = targetUserFeatures.dot(denseFeatures)
      val normA = Vectors.norm(targetUserFeatures, 2.0)
      val normB = Vectors.norm(denseFeatures, 2.0)
      dotProduct / (normA * normB)
    }

    // Compute cosine similarity for all users
    val usersSimilarity = dfWithDenseFeatures
      .filter($"user_id" =!= targetUser)
      .withColumn("cosine_sim", cosSimilarity(col("dense_features")))
      .select("user_id", "cosine_sim")
      .orderBy($"cosine_sim".desc)
      .limit(3)

    // Show top 3 similar users
    println("Top 3 similar users:")
    usersSimilarity.show()

    val tCosineSimilarityF = System.nanoTime()

    val tFinalRecommendI = System.nanoTime()

    // Extract games recommended by the target user
    val titlesPlayedByTargetUser = cleanMerge
      .filter(col("user_id") === targetUser)
      .select("title")
      .distinct()
      .as[String]
      .collect()

    // Extract relevant user IDs from recommendations
    val userIdsToFind = usersSimilarity.select("user_id").as[Int].collect.toSet

    // Filter datasetDF to remove already played games
    val finalRecommendations = cleanMerge
      .filter(col("user_id").isin(userIdsToFind.toArray: _*) &&
        !col("title").isin(titlesPlayedByTargetUser: _*) &&
        col("is_recommended") === true)
      .groupBy("app_id", "title")
      .agg(collect_list("user_id").alias("users"))

    val tFinalRecommendF = System.nanoTime()

    finalRecommendations.take(100).foreach(println)

    /*

    +-------+--------------------+-------------------+
| app_id|               title|              users|
+-------+--------------------+-------------------+
|1085660|           destiny 2|         [14044364]|
|1060670|      taboos: cracks|          [7889674]|
|1146630|      yokai's secret|          [7889674]|
| 307690|sleeping dogs: de...|         [14044364]|
|1267910|         melvor idle|         [14044364]|
|1509090|         seek girl ⅷ|          [7889674]|
|1126290|                lost|[7889674, 14044364]|
|1172470|       apex legends™|         [14044364]|
|1263370|     seek girl:fog ⅰ|          [7889674]|
|1205240|       tentacle girl|          [7889674]|
+-------+--------------------+-------------------+
     */

    // Calculating execution times
    println("\n\nExecution time(preprocessing):\t"+ (tPreProcessingF-tPreProcessingI)/1000000 + "ms\n")
    println("\n\nExecution time(Tf-Idf calculation):\t"+ (tTFIDFF-tTFIDFI)/1000000 + "ms\n")
    println("\n\nExecution time(Cosine similarity calculation):\t"+ (tCosineSimilarityF-tCosineSimilarityI)/1000000 + "ms\n")
    println("\n\nExecution time(final recommendation):\t"+ (tFinalRecommendF-tFinalRecommendI)/1000000 + "ms\n")
    println("\n\nExecution time(total):\t"+ (tFinalRecommendF-tPreProcessingI)/1000000 + "ms\n")

    spark.stop()
  }
}