package com.unibo.recommendationsystem
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, collect_list, count, udf}
import org.apache.spark.ml.linalg.Vector

import org.apache.spark.sql.functions._


object recommendationMLlib {

  def main(args: Array[String]): Unit = {

    //Initialize SparkSession
    val spark = SparkSession
      .builder
      .appName("recommendationsystem")
      .config("spark.master", "local[*]")
      .getOrCreate()

    val dataPathRec = "C:\\Users\\gbeks\\IdeaProjects\\recommendationsystem\\steam-datasets\\recommendations.csv"
    val dataPathGames = "C:\\Users\\gbeks\\IdeaProjects\\recommendationsystem\\steam-datasets\\games.csv"

//    val dataPathRec = "gs://dataproc-staging-us-central1-534461255477-conaqzw0/data/recommendations.csv"
//    val dataPathGames = "gs://dataproc-staging-us-central1-534461255477-conaqzw0/data/games.csv"

    val tPreProcessingI = System.nanoTime()

    //Load dataset as Dataframe
    val dfRec = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathRec)
    val dfGames = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathGames)

    //Select useful columns
    val selectedRec = dfRec.select("app_id", "user_id" , "is_recommended")
    val selectedGames = dfGames.select("app_id", "title")


    //Merge the dataframe for app_id
    val merged = selectedRec.join(selectedGames, Seq("app_id"), "inner")

    //Clean the dataset from useless whitespace
    val cleanMerge = merged
      .withColumn("title", lower(trim(regexp_replace(col("title"), "\\s+", " "))))

    /* Code to split the dataset for faster run
    val splits = cleanMerge.randomSplit(Array(0.7, 0.3), seed=123L) // 70/30 split
    val trainingDF = splits(0)
    val testingDF = splits(1)
    println("testingDF printschema")
    testingDF.printSchema()

     */

    //Tokenization of titles with MLlib class
    val tokenizer = new Tokenizer()
      .setInputCol("title")
      .setOutputCol("words")
    val tokenizedData = tokenizer.transform(cleanMerge)

    //tokenizedData.printSchema()
    //tokenizedData.show()

    //UserDefinedFunction for custom function
    def flattenWords = udf((s: Seq[Seq[String]]) => s.flatten) //The function flattens the nested sequence into a single list of strings, combining words from all the inner lists.

    //Aggregate tokenized data by user ID, then apply an aggregation function to each user's data.
    //Then, use the user-defined function flattenWords to flatten the resulting list of lists of words into a single list.
    val aggregateData = tokenizedData.groupBy("user_id").agg(flattenWords(collect_list("words").as("words")))

    //Filtering out all users with less than 20 words in their aggregated words list
    val filteredData = aggregateData.filter(count("words") >= 20)

    val tPreProcessingF = System.nanoTime()


    val tTFIDFI = System.nanoTime()

    //HashingTF converts a collection of words into a fixed-length feature vector
    val hashingTF = new HashingTF()
      .setInputCol("UDF(collect_list(words) AS words)")
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

   // rescaledData.printSchema()
   // rescaledData.show()

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

    /*
        val firstUserId = newDf.select("user_id").first().getAs[Int]("user_id")
        println("Target User: " + firstUserId)
     */



    //Select a user for the recommendation
    val targetUser = 2591067

    val id_list = Seq(targetUser)

    val t0 = System.nanoTime()

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
    val usersSimilar = drop.limit(3).select(col("user_id").alias("user"))
    //sersSimilar.collect().foreach(println)
    //println("Recommendations Top3")
    /*
    [6019065]
    [8605254]
    [6222146]
    Recommendations Top3
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


    /*
      +-------+--------------------+------------------+
      | app_id|               title|             users|
      +-------+--------------------+------------------+
      |1105670|      the last spell|         [6019065]|
      | 629760|             mordhau|         [6019065]|
      |1648470|                    |         [8605254]|
      |1500750|         tauren maze|         [8605254]|
      |1263370|        seek girlfog|[8605254, 6019065]|
      |1013130|  happy anime puzzle|         [8605254]|
      |1397350|  - long time no see|         [8605254]|
      |1148510|        pretty angel|[6019065, 6222146]|
      |1460040|        love fantasy|         [8605254]|
      |1153430|           love wish|[8605254, 6019065]|
      |1426110|love n dream virt...|         [8605254]|
      |1060670|       taboos cracks|[6222146, 6019065]|
      |1468160|        cube racer 2|         [6019065]|
      |1211360|            neomorph|[8605254, 6222146]|
      |1146630|       yokais secret|[6019065, 8605254]|
      | 355980|     dungeon warfare|         [6019065]|
      | 391220|rise of the tomb ...|         [8605254]|
      |1274610|leslove.club emil...|         [8605254]|
      |1182760|           starlight|         [8605254]|
      |1605010|      adorable witch|         [8605254]|
      +-------+--------------------+------------------+
      only showing top 20 rows
     */

    // Calculating execution times
    println("\n\nExecution time(preprocessing):\t"+ (tPreProcessingF-tPreProcessingI)/1000000 + "ms\n")
    println("\n\nExecution time(Tf-Idf calculation):\t"+ (tTFIDFF-tTFIDFI)/1000000 + "ms\n")
    println("\n\nExecution time(Cosine similarity calculation):\t"+ (tCosineSimilarityF-tCosineSimilarityI)/1000000 + "ms\n")
    println("\n\nExecution time(final recommendation):\t"+ (tFinalRecommendF-tFinalRecommendI)/1000000 + "ms\n")
    println("\n\nExecution time(total):\t"+ (tFinalRecommendF-tPreProcessingI)/1000000 + "ms\n") //da sistemare

    spark.stop()
  }
}
