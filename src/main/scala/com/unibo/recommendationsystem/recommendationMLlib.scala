package com.unibo.recommendationsystem
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
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

    val dataPathRec = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/recommendations.csv"
    val dataPathGames = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/games.csv"

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

    //This udf take a spark Ml Vector as input and convert it as a dense vector
    //Think of this like filling in any zero values with explicit zeros to form a dense array representation
    val asDense = udf((v: Vector) => v.toDense)

    val newDf = rescaledData
      .withColumn("dense_features", asDense(col("features")))

    //udf for calculate cosine similarity between targetUserVector and otherUserVector
    val cosSimilarity = udf { (x: Vector, y: Vector) =>
      val v1 = x.toArray
      val v2 = y.toArray
      val l1 = scala.math.sqrt(v1.map(x => x*x).sum) //Magnitude
      val l2 = scala.math.sqrt(v2.map(x => x*x).sum) //Magnitude
      val scalar = v1.zip(v2).map(p => p._1*p._2).sum //Dot Product
      scalar/(l1*l2)
    }


    //Select a user for the recommendation
    val targetUser = 2591067

    val id_list = Seq(targetUser)

    //Select target user features
    val filtered_df = newDf
      .filter(col("user_id").isin(id_list: _*))
      .select(col("user_id").as("id_frd"), col("dense_features").as("dense_frd"))//select

    val joinedDf = newDf.join(broadcast(filtered_df), //Create a broadcasted version of filtered_df. Broadcasting is a Spark optimization technique for handling joins with DataFrames of very different sizes
        col("user_id") =!= col("id_frd"))//To not compare targetUser with himself
      .withColumn("cosine_sim", cosSimilarity(col("dense_frd"), col("dense_features")))//add new column with the cosine similarity score

    val filtered = joinedDf
      .withColumn("cosine_sim", when(col("cosine_sim").isNaN, 0).otherwise(col("cosine_sim"))) //Replace Nan values with 0
      .orderBy(col("cosine_sim").desc) //sorting in descending order based on the "cosine_sim" column

    //Drop useless column
    val drop = filtered.drop("dense_features", "dense_frd", "hashedFeatures")

    //Find the three users most similar to the target user
    val usersSimilar = drop.limit(15).select(col("user_id").alias("user"))
    println("Top 15 similar user")
    usersSimilar.show()

    /*
    [6019065]
    [8605254]
    [6222146]
     */

    val tCosineSimilarityF = System.nanoTime()

    val tFinalRecommendI = System.nanoTime()

    //Find games played by users similar to the target user
    val resultDF = cleanMerge.join(usersSimilar, cleanMerge("user_id") === usersSimilar("user")).drop(col("user_id"))

    //Extract games recommended by the target user
    val userDF = cleanMerge.filter(cleanMerge("user_id").alias("target_user") === targetUser).toDF()

    val excludedGamesDF = resultDF.join(userDF, Seq("app_id"), "leftanti") //Finds all rows from resultDF (games played by similar users).
      //Excludes any rows from resultDF where the app_id appears in userDF (games already played by the target user).
      .select("app_id","title","user", "is_recommended")

    //Exclude games not recommended by the similar users ang group recommended app_id
    val aggregatedDF = excludedGamesDF
      .where(col("is_recommended") === true)
      .groupBy("app_id", "title")
      .agg(collect_list("user").alias("users"))

    val tFinalRecommendF = System.nanoTime()

    aggregatedDF.show()

    // Calculating execution times
    println("\n\nExecution time(preprocessing):\t"+ (tPreProcessingF-tPreProcessingI)/1000000 + "ms\n")
    println("\n\nExecution time(Tf-Idf calculation):\t"+ (tTFIDFF-tTFIDFI)/1000000 + "ms\n")
    println("\n\nExecution time(Cosine similarity calculation):\t"+ (tCosineSimilarityF-tCosineSimilarityI)/1000000 + "ms\n")
    println("\n\nExecution time(final recommendation):\t"+ (tFinalRecommendF-tFinalRecommendI)/1000000 + "ms\n")
    println("\n\nExecution time(total):\t"+ (tFinalRecommendF-tPreProcessingI)/1000000 + "ms\n") //da sistemare

    spark.stop()
  }
}