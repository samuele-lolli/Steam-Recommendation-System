package com.unibo.recommendationsystem

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.{ArrayType, BooleanType, DateType, DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.storage.StorageLevel

object tagMLlib {

  def main(args: Array[String]): Unit = {
    // Initialize SparkSession
    val spark = SparkSession.builder
      .appName("recommendationsystem")
      .config("spark.master", "local[*]")

      .getOrCreate()


    val recSchema = StructType(Array(
      StructField("app_id", IntegerType, true),
      StructField("helpful", IntegerType, true),
      StructField("funny", IntegerType, true),
      StructField("date", DateType, true),
      StructField("is_recommended", BooleanType, true),
      StructField("hours", DoubleType, true),
      StructField("user_id", IntegerType, true),
      StructField("review_id", IntegerType, true)
    ))

    // Schema for games.csv
    val gamesSchema = StructType(Array(
      StructField("app_id", IntegerType, true),
      StructField("title", StringType, true),
      StructField("date_release", DateType, true),
      StructField("win", BooleanType, true),
      StructField("mac", BooleanType, true),
      StructField("linux", BooleanType, true),
      StructField("rating", StringType, true),
      StructField("positive_ratio", IntegerType, true),
      StructField("user_reviews", IntegerType, true),
      StructField("price_final", DoubleType, true),
      StructField("price_original", DoubleType, true),
      StructField("discount", DoubleType, true),
      StructField("steam_deck", BooleanType, true)
    ))

    val metadataSchema = StructType(Array(
      StructField("app_id", IntegerType, nullable = false),
      StructField("description", StringType, nullable = true),
      StructField("tags", ArrayType(StringType), nullable = true) // Array di stringhe per i tag
    ))

    // Paths for datasets
    val dataPathRec = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/recommendations.csv"
    val dataPathGames = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/games.csv"
    val metadataPath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/games_metadata.json"

    val tPreProcessingI = System.nanoTime()

    // Load datasets
    val dfRec = spark.read.format("csv").option("header", "true").schema(recSchema).load(dataPathRec).filter("is_recommended = true")
    val dfGames = spark.read.format("csv").option("header", "true").schema(gamesSchema).load(dataPathGames)
    val dfMetadata = spark.read.format("json").schema(metadataSchema).load(metadataPath)


    // Select relevant columns and join datasets on app_id
    val selectedRec = dfRec.select("app_id", "user_id")
    val selectedGames = dfGames.select("app_id", "title")

    val merged = selectedRec.join(selectedGames, Seq("app_id"))
      .join(dfMetadata.drop("description"), Seq("app_id"))
      .filter(size(col("tags")) > 0)

    val cleanMerge = merged.withColumn("tags", transform(col("tags"), tag => lower(trim(regexp_replace(tag, "\\s+", " ")))))
      .withColumn("tagsString", concat_ws(",", col("tags"))).persist(StorageLevel.MEMORY_AND_DISK).drop("tags")


    // Tokenize the titles
  /*  val tokenizer = new Tokenizer().setInputCol("tagsString").setOutputCol("words")
    val tokenizedData = tokenizer.transform(cleanMerge)
   */

    val tokenizedData = cleanMerge
      .withColumn("words", split(col("tagsString"), ","))

    val aggregateData = tokenizedData.groupBy("user_id")
      .agg(flatten(collect_list("words")).as("words")).persist(StorageLevel.MEMORY_AND_DISK)

    //    val filteredData = aggregateData.filter(size(col("words")) >= 50)

    val tPreProcessingF = System.nanoTime()

    val tTFIDFI = System.nanoTime()

    import spark.implicits._
    // Convert words to feature vectors using HashingTF and IDF
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("hashedFeatures").setNumFeatures(20000)
    val featurizedData = hashingTF.transform(aggregateData).persist(StorageLevel.MEMORY_AND_DISK)
    val idf = new IDF().setInputCol("hashedFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)

    val tTFIDFF = System.nanoTime()

    val tCosineSimilarityI = System.nanoTime()

    // UDF to convert sparse vectors to dense vectors for cosine similarity
    val denseVector = udf { (v: Vector) => Vectors.dense(v.toArray) }
    val dfWithDenseFeatures = rescaledData.withColumn("dense_features", denseVector(col("features")))//.persist(StorageLevel.MEMORY_AND_DISK)

    // Target user ID
    val targetUser = 4893896

    def cosineSimilarity(targetVector: Vector): UserDefinedFunction = udf { (otherVector: Vector) =>

      val dotProduct = targetVector.dot(otherVector)
      val normA = Vectors.norm(targetVector, 2)
      val normB = Vectors.norm(otherVector, 2)
      dotProduct / (normA * normB)
    }

    // Assume rescaledData is the DataFrame containing user_id and features (of type Vector)
    val targetUserFeatures = dfWithDenseFeatures.filter($"user_id" === targetUser)
      .select("features").first().getAs[Vector]("features")


    val usersSimilarity = dfWithDenseFeatures
      .filter($"user_id" =!= targetUser)
      .withColumn("cosine_sim", cosineSimilarity(targetUserFeatures)(col("features")))
      .select("user_id", "cosine_sim")
      .orderBy($"cosine_sim".desc)
      .limit(3)


    usersSimilarity.take(3).foreach(println)
    println("Top 3 similar users:")
    /*
[8971360,0.8591792924376707]
[13271546,0.8443858670280873]
[11277999,0.8432720374293458]
     */


    val tCosineSimilarityF = System.nanoTime()
    val tFinalRecommendI = System.nanoTime()

    // Extract games recommended by the target user
    val titlesPlayedByTargetUser = cleanMerge.filter($"user_id" === targetUser)
      .select("tagsString").distinct().as[String].collect()

    // Extract relevant user IDs from recommendations
    val userIdsToFind = usersSimilarity.select("user_id").as[Int].collect.toSet

    // Filter dataset to remove already played games and aggregate recommendations
    val finalRecommendations = cleanMerge.filter(col("user_id").isin(userIdsToFind.toArray: _*)
        && !col("title").isin(titlesPlayedByTargetUser: _*))
      .groupBy("app_id", "title")
      .agg(collect_list("user_id").alias("users"))


    finalRecommendations.show(finalRecommendations.count.toInt, truncate = false)

    val tFinalRecommendF = System.nanoTime()

    // Calculate and display execution times
    println(s"\n\nExecution time (preprocessing): ${(tPreProcessingF - tPreProcessingI) / 1000000} ms")
    println(s"\n\nExecution time (Tf-Idf calculation): ${(tTFIDFF - tTFIDFI) / 1000000} ms")
    println(s"\n\nExecution time (cosine similarity calculation): ${(tCosineSimilarityF - tCosineSimilarityI) / 1000000} ms")
    println(s"\n\nExecution time (final recommendation): ${(tFinalRecommendF - tFinalRecommendI) / 1000000} ms")
    println(s"\n\nExecution time (total): ${(tFinalRecommendF - tPreProcessingI) / 1000000} ms")

    //LOCALE
    /*
+-------+-------------------------------------------------------+--------------------+
|app_id |title                                                  |users               |
+-------+-------------------------------------------------------+--------------------+
|35450  |Red Orchestra 2: Heroes of Stalingrad with Rising Storm|[8971360, 13271546] |
|1072040|Panzer Corps 2                                         |[11277999]          |
|4700   |Total War: MEDIEVAL II – Definitive Edition            |[8971360]           |
|304930 |Unturned                                               |[8971360]           |
|268400 |Panzer Corps Gold                                      |[11277999]          |
|555570 |Infestation: The New Z                                 |[13271546]          |
|34030  |Total War: NAPOLEON – Definitive Edition               |[13271546]          |
|41700  |S.T.A.L.K.E.R.: Call of Pripyat                        |[13271546]          |
|312450 |Order of Battle: World War II                          |[8971360]           |
|301640 |Zombie Army Trilogy                                    |[13271546]          |
|549080 |Tank Warfare: Tunisia 1943                             |[13271546]          |
|228200 |Company of Heroes                                      |[11277999]          |
|1369370|Combat Mission Shock Force 2                           |[13271546]          |
|736220 |Post Scriptum                                          |[13271546]          |
|494840 |UBOAT                                                  |[13271546]          |
|312980 |Graviteam Tactics: Mius-Front                          |[13271546]          |
|242860 |Verdun                                                 |[8971360, 13271546] |
|995460 |Miracle Snack Shop                                     |[13271546]          |
|447820 |Day of Infamy                                          |[13271546]          |
|302670 |Call to Arms                                           |[13271546]          |
|244450 |Men of War: Assault Squad 2                            |[11277999, 8971360] |
|418460 |Rising Storm 2: Vietnam                                |[8971360, 13271546] |
|1200   |Red Orchestra: Ostfront 41-45                          |[11277999, 13271546]|
|485980 |Syrian Warfare                                         |[13271546]          |
|201810 |Wolfenstein: The New Order                             |[13271546]          |
|572410 |Steel Division: Normandy 44                            |[13271546]          |
|231430 |Company of Heroes 2                                    |[13271546]          |
|350080 |Wolfenstein: The Old Blood                             |[13271546]          |
|222880 |Insurgency                                             |[13271546]          |
|712100 |A Total War Saga: THRONES OF BRITANNIA                 |[13271546]          |
|259080 |Just Cause 2: Multiplayer Mod                          |[11277999]          |
|251060 |Wargame: Red Dragon                                    |[13271546]          |
|363680 |Battlefleet Gothic: Armada                             |[13271546]          |
|1238060|Dead Space™ 3                                          |[13271546]          |
+-------+-------------------------------------------------------+--------------------+



Execution time (preprocessing): 2227 ms


Execution time (Tf-Idf calculation): 192343 ms


Execution time (cosine similarity calculation): 16422 ms


Execution time (final recommendation): 25750 ms


Execution time (total): 236744 ms
     */

    //CLUSTER-PRIMO
    /*
    [65064,0.8793965595063066]
[13498880,0.8783593100136686]
[7002264,0.8755346657835146]
Top 3 similar users:
[240,Counter-Strike: Source,WrappedArray(7002264)]
[32470,STAR WARS™ Empire at War - Gold Pack,WrappedArray(7002264)]
[33100,Alien Shooter,WrappedArray(13498880)]
[65980,Sid Meier's Civilization®: Beyond Earth™,WrappedArray(7002264)]
[102600,Orcs Must Die!,WrappedArray(13498880)]
[108710,Alan Wake,WrappedArray(65064)]
[219150,Hotline Miami,WrappedArray(7002264)]
[234270,Ken Follett's The Pillars of the Earth,WrappedArray(65064)]
[240720,Getting Over It with Bennett Foddy,WrappedArray(13498880)]
[257420,Serious Sam 4,WrappedArray(7002264)]
[258090,99 Spirits,WrappedArray(7002264)]
[282070,This War of Mine,WrappedArray(65064)]
[282800,100% Orange Juice,WrappedArray(7002264)]
[285190,Warhammer 40000: Dawn of War III,WrappedArray(7002264)]
[289130,ENDLESS™ Legend,WrappedArray(65064)]
[308040,Back to Bed,WrappedArray(7002264)]
[310950,Street Fighter V,WrappedArray(7002264, 65064)]
[333600,NEKOPARA Vol. 1,WrappedArray(13498880)]
[335190,200% Mixed Juice!,WrappedArray(7002264)]
[356400,Thumper,WrappedArray(13498880)]
[365450,Hacknet,WrappedArray(13498880)]
[367450,Poly Bridge,WrappedArray(13498880)]
[368360,60 Seconds!,WrappedArray(13498880)]
[372330,Beauty Bounce,WrappedArray(7002264)]
[382560,Hot Lava,WrappedArray(13498880)]
[384180,Prominence Poker,WrappedArray(13498880)]
[385800,NEKOPARA Vol. 0,WrappedArray(13498880)]
[400910,Rabi-Ribi,WrappedArray(13498880, 7002264)]
[404180,Club Life,WrappedArray(7002264)]
[407310,NEKO-NIN exHeart,WrappedArray(7002264)]
[407330,Sakura Dungeon,WrappedArray(7002264)]
[407980,Sakura Beach 2,WrappedArray(7002264)]
[420110,NEKOPARA Vol. 2,WrappedArray(13498880)]
[429580,A Wild Catgirl Appears!,WrappedArray(7002264)]
[434650,Lost Castle / 失落城堡,WrappedArray(13498880)]
[469820,Genital Jousting,WrappedArray(13498880)]
[470310,TROUBLESHOOTER: Abandoned Children,WrappedArray(65064)]
[474960,Quantum Break,WrappedArray(65064)]
[489020,iGrow Game,WrappedArray(7002264)]
[521540,This World Unknown,WrappedArray(7002264)]
[528230,SYNTHETIK: Legion Rising,WrappedArray(65064)]
[530890,Haydee,WrappedArray(13498880)]
[569800,LEAVES - The Journey,WrappedArray(13498880)]
[569810,LEAVES - The Return,WrappedArray(13498880)]
[577670,Demolish & Build 2018,WrappedArray(13498880)]
[601220,Zup! F,WrappedArray(7002264)]
[602520,NEKOPARA Vol. 3,WrappedArray(7002264, 13498880)]
[606800,Startup Company,WrappedArray(7002264)]
[606880,GreedFall,WrappedArray(65064)]
[627270,Injustice™ 2,WrappedArray(65064)]
[654820,Akin Vol 2,WrappedArray(13498880)]
[656350,UnderMine,WrappedArray(13498880)]
[673950,Farm Together,WrappedArray(13498880)]
[678950,DRAGON BALL FighterZ,WrappedArray(65064)]
[680420,OUTRIDERS,WrappedArray(65064)]
[691770,Eiyu*Senki – The World Conquest,WrappedArray(65064)]
[714010,Aimlabs,WrappedArray(13498880)]
[719040,Wasteland 3,WrappedArray(7002264)]
[728740,Sniper Elite V2 Remastered,WrappedArray(65064)]
[733740,Sakura Cupid,WrappedArray(7002264)]
[736190,Chinese Parents,WrappedArray(65064)]
[739080,9 Monkeys of Shaolin,WrappedArray(65064)]
[740130,Tales of Arise,WrappedArray(65064)]
[759440,Learn Japanese To Survive! Kanji Combat,WrappedArray(7002264)]
[760460,WEED,WrappedArray(7002264)]
[802870,The Ditzy Demons Are in Love With Me,WrappedArray(13498880)]
[810670,NALOGI,WrappedArray(7002264)]
[841350,NALOGI 2,WrappedArray(7002264)]
[867290,Crossroads Inn Anniversary Edition,WrappedArray(65064)]
[870780,Control Ultimate Edition,WrappedArray(65064)]
[872410,ROMANCE OF THE THREE KINGDOMS XIV,WrappedArray(65064)]
[882100,XCOM®: Chimera Squad,WrappedArray(65064)]
[882960,Visitor 来访者,WrappedArray(65064)]
[899970,NEKOPARA Extra,WrappedArray(7002264)]
[911430,Good Company,WrappedArray(7002264)]
[914710,Cat Quest II,WrappedArray(13498880)]
[949290,Winkeltje: The Little Shop,WrappedArray(65064)]
[960910,Heavy Rain,WrappedArray(65064)]
[960990,Beyond: Two Souls,WrappedArray(65064)]
[966330,Flower,WrappedArray(13498880)]
[970830,The Dungeon Of Naheulbeuk: The Amulet Of Chaos,WrappedArray(65064)]
[979690,The Ascent,WrappedArray(65064)]
[985760,3D Custom Lady Maker,WrappedArray(65064)]
[987840,Expeditions: Rome,WrappedArray(65064)]
[994730,Banner of the Maid,WrappedArray(65064)]
[995980,Fae Tactics,WrappedArray(65064)]
[997070,Marvel's Avengers - The Definitive Edition,WrappedArray(65064)]
[998930,Seek Girl,WrappedArray(65064)]
[1008710,Wet Girl,WrappedArray(13498880, 7002264, 65064)]
[1009290,SWORD ART ONLINE Alicization Lycoris,WrappedArray(7002264)]
[1012880,60 Seconds! Reatomized,WrappedArray(13498880)]
[1027620,The Language of Love,WrappedArray(7002264)]
[1058530,H-Rescue,WrappedArray(65064, 13498880)]
[1060670,Taboos: Cracks,WrappedArray(65064, 13498880)]
[1069740,Seen,WrappedArray(7002264)]
[1070330,Russian Life Simulator,WrappedArray(7002264)]
[1083210,符文女孩/Rune Girl,WrappedArray(65064)]
[1094730,Black Legend,WrappedArray(65064)]
[1096720,CATGIRL LOVER,WrappedArray(65064)]
[1101450,Miss Neko,WrappedArray(13498880, 7002264, 65064)]


Execution time (preprocessing): 4138 ms


Execution time (Tf-Idf calculation): 137719 ms


Execution time (cosine similarity calculation): 4806 ms


Execution time (final recommendation): 4703 ms


Execution time (total): 151368 ms
     */

    spark.stop()
  }
}
