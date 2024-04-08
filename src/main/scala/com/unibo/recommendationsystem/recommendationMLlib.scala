package com.unibo.recommendationsystem
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, collect_list, concat, count, lit, udf}
import org.apache.spark.ml.linalg.{SparseVector, Vector}

import org.apache.spark.sql.functions._



object recommendationMLlib {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("recommendationsystem")
      .config("spark.master", "local[*]")
      .getOrCreate()
    println("---------Initializing Spark-----------")

    //val dataPathRec = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/recommendations.csv"
    //val dataPathGames = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/games.csv"

    val dataPathRec = "gs://dataproc-staging-us-central1-534461255477-conaqzw0/data/recommendations.csv"

    val dataPathGames = "gs://dataproc-staging-us-central1-534461255477-conaqzw0/data/games.csv"



    val t4 = System.nanoTime()
//
    //val modelPath = "/Users/leonardovincenzi/IdeaProjects/scalaProject/out/results.txt"
    //val sc = spark.sparkContext

    val dfRec = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathRec)
    val dfGames = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathGames)


    val selectedRec = dfRec.select("app_id", "user_id" , "is_recommended")

    val selectedGames = dfGames.select("app_id", "title")



    val merged = selectedRec.join(selectedGames, Seq("app_id"), "inner")


    val cleanMerge = merged
      .withColumn("title", regexp_replace(col("title"), "[^\\w\\sа-яА-ЯЁё.,!?\\-]", ""))
      .withColumn("title", lower(trim(regexp_replace(col("title"), "\\s+", " "))))


    val t5 = System.nanoTime()

    /*val splits = cleanMerge.randomSplit(Array(0.7, 0.3), seed=123L) // 70/30 split
    val trainingDF = splits(0)
    val testingDF = splits(1)
    println("testingDF printschema")
    testingDF.printSchema()

     */
    val t2 = System.nanoTime()


    // Tokenization
    val tokenizer = new Tokenizer()
      .setInputCol("title")
      .setOutputCol("words")
    val tokenizedData = tokenizer.transform(cleanMerge)

    //tokenizedData.printSchema()
    //tokenizedData.show()

    def flattenWords = udf((s: Seq[Seq[String]]) => s.flatten)


    val aggregateData = tokenizedData.groupBy("user_id").agg(flattenWords(collect_list("words").as("words")))

    val filteredData = aggregateData.filter(count("words") >= 20)


    // HashingTF

    val hashingTF = new HashingTF()
      .setInputCol("UDF(collect_list(words) AS words)")
      .setOutputCol("hashedFeatures")
      .setNumFeatures(20000) // Adjust numFeatures as needed
    val featurizedData = hashingTF.transform(filteredData)


    // IDF
    val idf = new IDF().setInputCol("hashedFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)

    val t3 = System.nanoTime()

   // rescaledData.printSchema()
   // rescaledData.show()

    val asDense = udf((v: Vector) => v.toDense)

    val newDf = rescaledData
      .withColumn("dense_features", asDense(col("features")))




    val cosSimilarity = udf { (x: Vector, y: Vector) =>
      val v1 = x.toArray
      val v2 = y.toArray
      val l1 = scala.math.sqrt(v1.map(x => x*x).sum)
      val l2 = scala.math.sqrt(v2.map(x => x*x).sum)
      val scalar = v1.zip(v2).map(p => p._1*p._2).sum
      scalar/(l1*l2)
    }

    /*
        val firstUserId = newDf.select("user_id").first().getAs[Int]("user_id")
        println("Target User: " + firstUserId)
     */

    val firstUserId = 2591067

    val id_list = Seq(firstUserId)

    val t0 = System.nanoTime()


    val filtered_df = newDf
      .filter(col("user_id").isin(id_list: _*))
      .select(col("user_id").as("id_frd"), col("dense_features").as("dense_frd"))


    val joinedDf = newDf.join(broadcast(filtered_df),
        col("user_id") =!= col("id_frd"))
      .withColumn("cosine_sim", cosSimilarity(col("dense_frd"), col("dense_features")))

    val filtered = joinedDf
      .withColumn("cosine_sim", when(col("cosine_sim").isNaN, 0).otherwise(col("cosine_sim")))
      .orderBy(col("cosine_sim").desc)
    //.filter(col("rank")between(2,11))


    val drop = filtered.drop("dense_features", "dense_frd", "hashedFeatures")

   // drop.show()



    val usersSimilar = drop.limit(3).select(col("user_id").alias("user"))
    usersSimilar.collect().foreach(println)
    println("Recommendations Top3")
    /*
[6019065]
[8605254]
[6222146]
Recommendations Top3
     */


    val resultDF = cleanMerge.join(usersSimilar, cleanMerge("user_id") === usersSimilar("user")).drop(col("user_id"))

    val userDF = cleanMerge.filter(cleanMerge("user_id").alias("target_user") === firstUserId).toDF()


    val excludedGamesDF = resultDF.join(userDF, Seq("app_id"), "leftanti")
      .select("app_id","title","user", "is_recommended")


    val aggregatedDF = excludedGamesDF
      .where(col("is_recommended") === true)
      .groupBy("app_id", "title")
      .agg(collect_list("user").alias("users"))

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

    val t1 = System.nanoTime()

    println("\n\nExecution time(Recommendation with Cosine Similarity):\t"+ (t1-t0)/1000000 + "ms\n")
    println("\n\nExecution time(Tf-Idf calculation):\t"+ (t3-t2)/1000000 + "ms\n")
    println("\n\nExecution time(preprocessing):\t"+ (t5-t4)/1000000 + "ms\n")
  }
}
