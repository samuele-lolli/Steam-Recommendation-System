package com.unibo.recommendationsystem.recommender

import com.unibo.recommendationsystem.utils.timeUtils
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.storage.StorageLevel

import scala.collection.Map

class sqlRecommendationV2 (spark: SparkSession, dataRec: Dataset[Row], dataGames: DataFrame, metadata: DataFrame) {

  /**
   * (SQL + ScalaCollection.Map version) Generates personalized recommendations for a target user
   *
   * @param targetUser The ID of the user for which we are generating recommendations
   */
  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (explodedDF, filteredData, gamesTitles, userGamesData) = timeUtils.time(preprocessData(), "Preprocessing Data", "SQL_HYBRID")
    println("Calculate term frequency and inverse document frequency...")
    val tfidfValues = timeUtils.time(calculateTFIDF(explodedDF, filteredData), "Calculating TF-IDF", "SQL_HYBRID")
    println("Calculate cosine similarity to get similar users...")
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(tfidfValues, targetUser), "Getting Similar Users", "SQL_HYBRID")
    println("Calculate final recommendation...")
    timeUtils.time(generateFinalRecommendations(topUsersSimilarity, targetUser, gamesTitles, userGamesData), "Generating Recommendations", "SQL_HYBRID")

    explodedDF.unpersist()
    filteredData.unpersist()
    tfidfValues.unpersist()
    gamesTitles.unpersist()
    userGamesData.unpersist()
  }

  /**
   * Preprocesses the input data to create intermediate dataframes needed for further calculations
   *
   * @return A tuple containing:
   *         - explodedDF: DataFrame with individual users and words (tags)
   *         - filteredData: DataFrame with aggregated tags for each user
   *         - gamesTitles: DataFrame with game IDs and titles
   *         - userGamesData: Complete, cleaned, and merged dataset
   */
  private def preprocessData(): (DataFrame, DataFrame, DataFrame, DataFrame) = {

    val selectedRec = dataRec.select("app_id", "user_id")
    val selectedGames = dataGames.select("app_id", "title")

    //Join datasets and remove games with zero tags
    val userGamesData = selectedRec.join(selectedGames, Seq("app_id"))
      .join(metadata.drop("description"), Seq("app_id"))
      .filter(size(col("tags")) > 0)

    //Clean and transform tags
    val userGamePairs = userGamesData
      .withColumn("tags", transform(col("tags"), tag => lower(trim(regexp_replace(tag, "\\s+", " ")))))
      .withColumn("tagsString", concat_ws(",", col("tags"))) // Join tags with commas
      .drop("tags")
      //.persist(StorageLevel.MEMORY_AND_DISK)


    //Create a list of tags for each single user
    val filteredData = userGamePairs
      .withColumn("words", split(col("tagsString"), ",")) // Split tags by commas
      .groupBy("user_id")
      .agg(flatten(collect_list("words")).as("words"))
      .persist(StorageLevel.MEMORY_AND_DISK)

    //Explode tags to calculate TF-IDF
    val explodedDF = filteredData.withColumn("word", explode(col("words"))).select("user_id", "word")
      .persist(StorageLevel.MEMORY_AND_DISK)

    val gamesTitles = dataGames.select("app_id", "title")

    (explodedDF, filteredData, gamesTitles, userGamesData)
  }

  /*
   * explodedDF
   * [53,world war ii]
   * [53,strategy]
   * [53,rts]
   */

  /*
   * filteredData
   * [53,WrappedArray(world war ii, strategy, rts, military, simulation, action, tactical, multiplayer, war, wargame, singleplayer, real time tactics, historical, realistic, open world, co-op, real-time, tanks, real-time with pause, replay value)]
   * [85,WrappedArray(action, warhammer 40k, co-op, fps, shooter, adventure, online co-op, multiplayer, first-person, violent, atmospheric, sci-fi, gore, singleplayer, horror, games workshop, mature, space, aliens, free to play, superhero, mmorpg, massively multiplayer, character customization, action, comic book, rpg, adventure, controller, third person, multiplayer, combat, beat 'em up, open world, exploration, pvp, fighting, online co-op, co-op)]
   * [133,WrappedArray(stealth, action, co-op, third person, multiplayer, singleplayer, shooter, adventure, tactical, third-person shooter, online co-op, parkour, fps, story rich, controller, strategy, atmospheric, first-person, rpg, mature, racing, combat racing, automobile sim, destruction, multiplayer, vehicular combat, great soundtrack, classic, action, singleplayer, music, driving, arcade, physics, simulation, local multiplayer, atmospheric, funny, comedy, casual)]
   */

  /*
   * gamesTitles
   * [13500,Prince of Persia: Warrior Within™]
   * [22364,BRINK: Agents of Change]
   * [113020,Monaco: What's Yours Is Mine]
   */

  /*
   * userGamesData
   * [3490,6215812,Venice Deluxe,WrappedArray(Casual)]
   * [3490,2113752,Venice Deluxe,WrappedArray(Casual)]
   * [4900,12062841,Zen of Sudoku,WrappedArray(Casual, Indie, Puzzle, Free to Play)]
   */

  /**
   * Computes TF-IDF values for all users based on their tags
   *
   * @param explodedDF   DataFrame with user-word pairs
   * @param filteredData DataFrame with user-wise aggregated tags as arrays
   * @return DataFrame containing TF-IDF scores for each user and word
   */
  private def calculateTFIDF(explodedDF: DataFrame, filteredData: DataFrame): DataFrame = {
    val wordsPerUser = explodedDF.groupBy("user_id").agg(count("*").alias("total_words"))

    //TF
    val tf = explodedDF.groupBy("user_id", "word")
      .agg(count("*").alias("term_count"))
      .join(wordsPerUser, "user_id")
      .withColumn("term_frequency", col("term_count") / col("total_words"))

    //DF
    val dfDF = explodedDF.groupBy("word")
      .agg(countDistinct("user_id").alias("document_frequency"))

    //Users
    val totalDocs = filteredData.count()

    //IDF
    dfDF.createOrReplaceTempView("dfDF")
    val idfDF = spark.sql(s"""SELECT word, log($totalDocs / document_frequency) AS idf FROM dfDF""")

    //TFIDF
    val tfidfDF = tf.join(idfDF, "word")
      .withColumn("tf_idf", col("term_frequency") * col("idf"))
      .select("user_id", "word", "tf_idf")

    tfidfDF
  }
  /*
   * tfidfDF
   * [392,anime,0.08031574163308791]
   * [1645,anime,0.02677191387769597]
   * [1699,anime,0.006490160940047508]
   */

  /**
   * Computes cosine similarity between the target user and all other users
   *
   * @param tfidfDF DataFrame with TF-IDF scores for each user and word
   * @param targetUser The ID of the target user
   * @return List of user IDs with the highest similarity
   */
  private def computeCosineSimilarity(tfidfDF: DataFrame, targetUser: Int): List[Int] = {

    def calculateCosineSimilarity(vector1: Map[String, Double], vector2: Map[String, Double], dotProductFunc: (Map[String, Double], Map[String, Double]) => Double, magnitudeFunc: Map[String, Double] => Double): Double = {
      val magnitude1 = magnitudeFunc(vector1)
      val magnitude2 = magnitudeFunc(vector2)
      if (magnitude1 == 0.0 || magnitude2 == 0.0) 0.0 else dotProductFunc(vector1, vector2) / (magnitude1 * magnitude2)
    }

    def dotProduct = (v1: Map[String, Double], v2: Map[String, Double]) => {
      v1.foldLeft(0.0) { case (acc, (key, value)) => acc + v2.getOrElse(key, 0.0) * value }
    }

    def magnitude = (vector: Map[String, Double]) => math.sqrt(vector.values.map(value => value * value).sum)

    def rowToTfIdfMap(row: Row): Map[String, Double] = {
      row.getAs[Seq[Row]]("tags").map(tag => tag.getString(0) -> tag.getDouble(1)).toMap
    }

    // Extract target user vector
    val targetUserData = tfidfDF.filter(col("user_id") === targetUser)
      .groupBy("user_id")
      .agg(collect_list(struct("word", "tf_idf")).alias("tags"))

    val targetUserVector = targetUserData.collect().headOption.map(rowToTfIdfMap).getOrElse(Map.empty[String, Double])

    // Compute similarity for other users
    val otherUsersData = tfidfDF.filter(col("user_id") =!= targetUser)
      .groupBy("user_id")
      .agg(collect_list(struct("word", "tf_idf")).alias("tags"))

    import org.apache.spark.sql.Encoders
    implicit val tupleEncoder: Encoder[(Int, Double)] = Encoders.product[(Int, Double)]

    val otherUsersWithSimilarity = otherUsersData.map { row =>
      val userId = row.getAs[Int]("user_id")
      val userVector = rowToTfIdfMap(row)
      val cosineSimilarity = calculateCosineSimilarity(targetUserVector, userVector, dotProduct, magnitude)
      (userId, cosineSimilarity)
    }.toDF("user_id", "cosine_similarity")

    val top3Users = otherUsersWithSimilarity.orderBy(desc("cosine_similarity")).limit(3)
    top3Users.select("user_id").collect().map(row => row.getAs[Int]("user_id")).toList
  }

  /*
   * top3Users
   * 8971360
   * 11277999
   * 9911449
   */

  /**
   * Generates and prints final game recommendations for a target user based on games played by similar users
   *
   * @param top3Users  List of top similar user IDs
   * @param targetUser The ID of the target user
   * @param gamesTitles DataFrame containing game IDs and titles
   * @param userGamesData Fully joined and cleaned data set
   */
  def generateFinalRecommendations(top3Users: List[Int], targetUser: Int, gamesTitles: DataFrame, userGamesData: DataFrame): Unit = {
    val gamesByTopUsers = userGamesData.filter(col("user_id").isin(top3Users: _*))
      .select("app_id", "user_id")

    val gamesByTargetUser = userGamesData.filter(col("user_id") === targetUser)
      .select("app_id")

    val recommendedGames = gamesByTopUsers.join(gamesByTargetUser, Seq("app_id"), "left_anti")

    val finalRecommendations = recommendedGames
      .join(gamesTitles.select("app_id", "title"), Seq("app_id"))
      .select("app_id", "title", "user_id")

    val groupedRecommendations = finalRecommendations
      .groupBy("app_id", "title")
      .agg(collect_list("user_id").alias("user_ids"))

    groupedRecommendations.show(groupedRecommendations.count.toInt, truncate = false)
  }
}
