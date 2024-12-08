package com.unibo.recommendationsystem.recommender

import com.unibo.recommendationsystem.utils.timeUtils
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

class mllibRecommendation(spark: SparkSession, dataRec: Dataset[Row], dataGames: DataFrame, metadata: DataFrame) {

  /**
   * (MLlib SQL version) Generates personalized recommendations for a target user
   *
   * @param targetUser Int, The ID of the user for which we are generating recommendations
   *
   */
  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (aggregateData, userGamePairs) = timeUtils.time(preprocessData(), "Preprocessing Data", "MlLib")
    println("Calculate term frequency and inverse document frequency...")
    val tfidfValues = timeUtils.time(calculateTFIDF(aggregateData), "Calculating TF-IDF", "MlLib")
    println("Calculate cosine similarity to get similar users...")
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(tfidfValues, targetUser), "Getting Similar Users", "MlLib")
    println("Calculate final recommendation...")
    timeUtils.time(generateFinalRecommendations(userGamePairs, topUsersSimilarity, targetUser), "Generating Recommendations", "MlLib")

    aggregateData.unpersist()
    userGamePairs.unpersist()
    tfidfValues.unpersist()
    topUsersSimilarity.unpersist()
  }

  /**
   * Preprocesses the input data to create intermediate dataframes needed for further calculations.
   *
   * @return A tuple of:
   *         - [Int, WrappedArray(String)] that maps each user with their tags for TF-IDF calculation
   *         - [Int, Int, String, String, ...] that contains game/user associations, game title and its relative tags
   *
   */
  private def preprocessData(): (DataFrame, DataFrame) = {
    val selectedRec = dataRec.select("app_id", "user_id")
    val selectedGames = dataGames.select("app_id", "title")

    val userGamesData = selectedRec
      .join(selectedGames, Seq("app_id"))
      .join(metadata.drop("description"), Seq("app_id"))
      .filter(size(col("tags")) > 0)

    val userGamePairs = userGamesData
      .withColumn("tags", transform(col("tags"), tag => lower(trim(regexp_replace(tag, "\\s+", " ")))))
      .withColumn("tagsString", concat_ws(",", col("tags")))
      .drop("tags")
      .persist(StorageLevel.MEMORY_AND_DISK)
    //.cache()

    val ugpTokenized = userGamePairs.withColumn("words", split(col("tagsString"), ","))

    val aggregateData = ugpTokenized
      .groupBy("user_id")
      .agg(flatten(collect_list("words")).as("words"))
      .persist(StorageLevel.MEMORY_AND_DISK)
    //.cache()

    (aggregateData, userGamePairs)
  }
    /*
     * userGamesData
     * [4900,12062841,Zen of Sudoku,casual,indie,puzzle,free to play]
     * [4900,10893520,Zen of Sudoku,casual,indie,puzzle,free to play]
     * [4900,10243247,Zen of Sudoku,casual,indie,puzzle,free to play]
     *
     * aggregateData
     * [463,WrappedArray(puzzle, casual, indie, 2d, physics, relaxing, singleplayer, minimalist, short, fast-paced, cute, trading card game, strategy, logic, psychological horror, difficult, action, education, horror, beautiful, psychological horror, multiplayer, free to play, battle royale, pvp, action, first-person, parkour, 3d, fps, platformer, arcade, physics, combat, casual, nudity, runner, racing, 3d platformer, sci-fi)]
     * [1088,WrappedArray(adventure, action, female protagonist, third person, singleplayer, story rich, third-person shooter, multiplayer, exploration, action-adventure, quick-time events, atmospheric, shooter, puzzle, stealth, cinematic, platformer, rpg, reboot, 3d vision)]
     * [1591,WrappedArray(free to play, horror, multiplayer, first-person, co-op, survival horror, shooter, online co-op, action, fps, memes, sci-fi, survival, psychological horror, atmospheric, strategy, difficult, indie, adventure, fantasy)]
     *

    */

  /**
   * Computes TF-IDF values for all users based on their tags
   *
   * @param aggregateData that contains each user and their tags with associated TF-IDF values
   * @return DataFrame containing TF-IDF values for each tag and user
   *
   */
  private def calculateTFIDF(aggregateData: DataFrame): DataFrame = {
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("hashedFeatures").setNumFeatures(20000)
    val featurizedData = hashingTF.transform(aggregateData).persist(StorageLevel.MEMORY_AND_DISK)//.cache()

    val idf = new IDF().setInputCol("hashedFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    idfModel.transform(featurizedData)
  }
  /*
   * [463,WrappedArray(puzzle, casual, indie, 2d, physics, relaxing, singleplayer, minimalist, short, fast-paced, cute, trading card game, strategy, logic, psychological horror, difficult, action, education, horror, beautiful, psychological horror, multiplayer, free to play, battle royale, pvp, action, first-person, parkour, 3d, fps, platformer, arcade, physics, combat, casual, nudity, runner, racing, 3d platformer, sci-fi),(20000,[349,776,2291,2768,4599,5049,5530,5566,5966,6230,7421,7548,7845,8023,8218,8250,9170,9341,9770,10405,11262,11273,11440,11712,11956,12111,13751,14055,14142,14276,14443,14942,16633,17031,17798,18809],[1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0]),(20000,[349,776,2291,2768,4599,5049,5530,5566,5966,6230,7421,7548,7845,8023,8218,8250,9170,9341,9770,10405,11262,11273,11440,11712,11956,12111,13751,14055,14142,14276,14443,14942,16633,17031,17798,18809],[1.7302931578817133,3.2127443925768318,0.8394589959004939,3.010483459454005,1.055022232707697,2.4180155088599298,3.3222262929471063,1.2988803766664814,1.3305530075463046,2.162867319853891,1.8442557278718714,1.3786699662036868,1.2280896704098183,4.419749903150982,3.7323474712556974,1.8422450147669338,0.8307478616653239,3.665163626523716,0.15870374729806824,2.4407820725061584,2.379376998652651,2.7517190390805983,0.5036136011040446,1.1331072326269802,2.949945816108345,2.160505153283781,1.0733073796519677,1.713318878009665,2.3190328451378446,3.753340030663497,2.797025744413799,1.3422542741250956,0.4956754550369963,1.950770095402198,0.3752726789656653,1.8245946488850548])]
   * [1088,WrappedArray(adventure, action, female protagonist, third person, singleplayer, story rich, third-person shooter, multiplayer, exploration, action-adventure, quick-time events, atmospheric, shooter, puzzle, stealth, cinematic, platformer, rpg, reboot, 3d vision),(20000,[731,2833,4051,4285,5234,5423,6216,6420,6696,7548,9770,9842,10957,11688,12640,14055,15464,16633,17715,17798],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]),(20000,[731,2833,4051,4285,5234,5423,6216,6420,6696,7548,9770,9842,10957,11688,12640,14055,15464,16633,17715,17798],[0.7941703789864701,0.35194405919755334,2.608304476842931,2.9429750381337834,0.8652566076803664,1.0657132906941778,3.06783780516898,1.7675743801929877,1.5040395028807223,1.3786699662036868,0.15870374729806824,3.8974589905293526,1.3313823365419577,0.6034520545039481,1.8734875151812935,1.713318878009665,1.8486268618914479,0.24783772751849814,0.8771216603091817,0.3752726789656653])]
   * [1591,WrappedArray(free to play, horror, multiplayer, first-person, co-op, survival horror, shooter, online co-op, action, fps, memes, sci-fi, survival, psychological horror, atmospheric, strategy, difficult, indie, adventure, fantasy),(20000,[776,2291,2833,2840,3859,5566,5966,7845,9170,11440,11688,11712,12813,13751,14431,14773,14914,16633,17715,17798],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]),(20000,[776,2291,2833,2840,3859,5566,5966,7845,9170,11440,11688,11712,12813,13751,14431,14773,14914,16633,17715,17798],[1.6063721962884159,0.8394589959004939,0.35194405919755334,1.9484240635796308,1.3023162055104078,1.2988803766664814,1.3305530075463046,1.2280896704098183,0.8307478616653239,0.5036136011040446,0.6034520545039481,1.1331072326269802,1.109334509076732,1.0733073796519677,1.1841472207416648,1.9276226095939772,0.7705348574053624,0.24783772751849814,0.8771216603091817,0.3752726789656653])]
   */


  /**
   * Computes cosine similarity between the target user and all other users
   *
   * @param rescaledData Map[Int, Map[String, Double] ], tf-idf score map for each userId
   * @param targetUser Int, the ID of the target user
   * @return A list of the top 3 most similar user IDs
   *
   */
  private def computeCosineSimilarity(rescaledData: DataFrame, targetUser: Int): DataFrame = {
    import spark.implicits._

    val targetUserFeatures = rescaledData
      .filter($"user_id" === targetUser)
      .select("features")
      .first()
      .getAs[Vector]("features")

    val cosineSimilarity = udf { (otherVector: Vector) =>
      val dotProduct = targetUserFeatures.dot(otherVector)
      val normA = Vectors.norm(targetUserFeatures, 2)
      val normB = Vectors.norm(otherVector, 2)
      dotProduct / (normA * normB)
    }

    rescaledData
      .filter($"user_id" =!= targetUser)
      .withColumn("cosine_sim", cosineSimilarity(col("features")))
      .select("user_id", "cosine_sim")
      .orderBy($"cosine_sim".desc)
      .limit(3)
 }

  /*
   *
   * [8971360,0.8591792924376707]
   * [13271546,0.8443858670280873]
   * [11277999,0.8432720374293458]
   *
   */

  /**
   * Generates and prints final game recommendations for a target user based on games played by similar users
   *
   * @param userGamePairs [Int, Int, String, String, ...] that contains game/user associations, game title and tags
   * @param usersSimilarity List[Int], list of IDs of the most similar users. 3 items
   * @param targetUser Int, the ID of the target user
   *
   */
  private def generateFinalRecommendations(userGamePairs: DataFrame, usersSimilarity: DataFrame, targetUser: Int): Unit = {
    import spark.implicits._

    val titlesPlayedByTargetUser = userGamePairs
      .filter($"user_id" === targetUser)
      .select("title")
      .distinct()
      .as[String]
      .collect()

    val userIdsToFind = usersSimilarity
      .select("user_id")
      .as[Int]
      .collect()
      .toSet

    val finalRecommendations = userGamePairs
      .filter(col("user_id").isin(userIdsToFind.toSeq: _*) && !col("title").isin(titlesPlayedByTargetUser: _*))
      .groupBy("app_id", "title")
      .agg(collect_list("user_id").alias("users"))

    finalRecommendations.show(finalRecommendations.count().toInt ,truncate = false)
  }
}