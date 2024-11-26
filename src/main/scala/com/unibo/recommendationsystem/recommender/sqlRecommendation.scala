package com.unibo.recommendationsystem.recommender

import com.unibo.recommendationsystem.utils.timeUtils
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

class sqlRecommendation(spark: SparkSession, dataRec: Dataset[Row], dataGames: DataFrame, metadata: DataFrame) {

  /**
   * Computes TF-IDF values for all users based on their tags
   *
   * @param targetUser ID of the user for whom recommendations are generated.
   */
  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (explodedDF, filteredData, gamesTitles, userGamesData) = timeUtils.time(preprocessData(), "Preprocessing Data", "SQL_FULL")

    println("Calculating term frequency and inverse document frequency...")
    val tfidfValues = timeUtils.time(calculateTFIDF(explodedDF, filteredData), "Calculating TF-IDF", "SQL_FULL")

    println("Calculating cosine similarity to get similar users...")
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(tfidfValues, targetUser), "Getting Similar Users", "SQL_FULL")

    println("Calculating final recommendation...")
    timeUtils.time(generateFinalRecommendations(topUsersSimilarity, targetUser, gamesTitles, userGamesData), "Generating Recommendations", "SQL_FULL")
    filteredData.unpersist()
    explodedDF.unpersist()
    tfidfValues.unpersist()
    userGamesData.unpersist()
  }


  /**
   * Preprocesses the input data to create intermediate dataframes needed for further calculations.
   *
   * @return A tuple containing:
   *         - explodedDF: DataFrame with individual users and words (tags).
   *         - filteredData: DataFrame with aggregated tags for each user.
   *         - gamesTitles: DataFrame with game IDs and titles.
   *         - userGamesData: Complete, cleaned, and merged dataset.
   */
  private def preprocessData(): (DataFrame, DataFrame, DataFrame, DataFrame) = {
    val selectedRec = dataRec.select("app_id", "user_id")
    val selectedGames = dataGames.select("app_id", "title")

    // Join datasets and remove untagged games
    val userGamesData = selectedRec.join(selectedGames, Seq("app_id"))
      .join(metadata.drop("description"), Seq("app_id"))
      .filter(size(col("tags")) > 0)

    // Clean and transform tags
    val cleanMerge = userGamesData
      .withColumn("tags", transform(col("tags"), tag => lower(trim(regexp_replace(tag, "\\s+", " ")))))
      .withColumn("tagsString", concat_ws(",", col("tags")))
      .drop("tags")

    // Create a list of tags for each single user
    val filteredData = cleanMerge
      .withColumn("words", split(col("tagsString"), ",")) // Splits tags by commas
      .groupBy("user_id")
      .agg(flatten(collect_list("words")).as("words"))

    // Explode tags for calculating TF-IDF
    val explodedDF = filteredData
      .withColumn("word", explode(col("words")))
      .select("user_id", "word")
      .persist(StorageLevel.MEMORY_AND_DISK)

    val gamesTitles = dataGames.select("app_id", "title")

    (explodedDF, filteredData, gamesTitles, userGamesData)
  }

  /*
   * explodedDF
   * [463,puzzle]
   * [463,casual]
   * [463,indie]
   * [463,2d]
   * [463,physics]
   */

  /*
   * filteredData
   * [53,WrappedArray(world war ii, strategy, rts, military, simulation, action, tactical, multiplayer, war, wargame, singleplayer, real time tactics, historical, realistic, open world, co-op, real-time, tanks, real-time with pause, replay value)]
   * [85,WrappedArray(action, warhammer 40k, co-op, fps, shooter, adventure, online co-op, multiplayer, first-person, violent, atmospheric, sci-fi, gore, singleplayer, horror, games workshop, mature, space, aliens, free to play, superhero, mmorpg, massively multiplayer, character customization, action, comic book, rpg, adventure, controller, third person, multiplayer, combat, beat 'em up, open world, exploration, pvp, fighting, online co-op, co-op)]
   * [133,WrappedArray(stealth, action, co-op, third person, multiplayer, singleplayer, shooter, adventure, tactical, third-person shooter, online co-op, parkour, fps, story rich, controller, strategy, atmospheric, first-person, rpg, mature, racing, combat racing, automobile sim, destruction, multiplayer, vehicular combat, great soundtrack, classic, action, singleplayer, music, driving, arcade, physics, simulation, local multiplayer, atmospheric, funny, comedy, casual)]
   * [243,WrappedArray(atmospheric, psychological, female protagonist, story rich, mythology, singleplayer, dark, adventure, horror, third person, dark fantasy, action, violent, indie, hack and slash, walking simulator, action-adventure, fantasy, realistic, blood)]
   */

  /*
   * gamesTitles
   * [13500,Prince of Persia: Warrior Within™]
   * [22364,BRINK: Agents of Change]
   * [113020,Monaco: What's Yours Is Mine]
   * [226560,Escape Dead Island]
   * [249050,Dungeon of the ENDLESS™]
   */

  /*
   * userGamesData
   * [3490,6215812,Venice Deluxe,WrappedArray(Casual)]
   * [3490,2113752,Venice Deluxe,WrappedArray(Casual)]
   * [4900,12062841,Zen of Sudoku,WrappedArray(Casual, Indie, Puzzle, Free to Play)]
   * [4900,10893520,Zen of Sudoku,WrappedArray(Casual, Indie, Puzzle, Free to Play)]
   * [4900,10243247,Zen of Sudoku,WrappedArray(Casual, Indie, Puzzle, Free to Play)]
   */

  /**
   * Computes TF-IDF values for all users based on their tags
   *
   * @param explodedDF   DataFrame with user-word pairs.
   * @param filteredData DataFrame with aggregated tags for each user.
   * @return DataFrame containing TF-IDF values for each word and user.
   */
  private def calculateTFIDF(explodedDF: DataFrame, filteredData: DataFrame): DataFrame = {
    val wordsPerUser = explodedDF.groupBy("user_id").agg(count("*").alias("total_words"))

    // Term Frequency (TF) calculation
    val tf = explodedDF.groupBy("user_id", "word")
      .agg(count("*").alias("term_count"))
      .join(wordsPerUser, "user_id")
      .withColumn("term_frequency", col("term_count") / col("total_words"))

    // Document Frequency (DF) calculatiom
    val dfDF = explodedDF.groupBy("word")
      .agg(countDistinct("user_id").alias("document_frequency"))

    // Total number of users (documents of TF-IDF calculation)
    val totalDocs = filteredData.count()

    // Calculate Inverse Document Frequency (IDF)
    val idfDF = dfDF.withColumn("idf", log(lit(totalDocs) / col("document_frequency")))

    // Combine TF and IDF to compute TF-IDF
    val tfidfValues = tf.join(idfDF, "word")
      .withColumn("tf_idf", col("term_frequency") * col("idf"))
      .select("user_id", "word", "tf_idf")

    tfidfValues
  }

  /*
   * tfidfValues
   * [1645,anime,0.02677191387769597]
   * [2122,anime,0.08031574163308791]
   * [2866,anime,0.08031574163308791]
   * [6658,anime,0.08031574163308791]
   * [7880,anime,0.08031574163308791]
   */


  /**
   * Computes cosine similarity between the target user and all other users
   *
   * @param tfidfDF    DataFrame with TF-IDF values for each word and user.
   * @param targetUser ID of the target user.
   * @return List of IDs of the most similar users.
   */
  private def computeCosineSimilarity(tfidfDF: DataFrame, targetUser: Int): List[Int] = {
    import spark.implicits._

    // Take the target vector by filtering out target's TF-IDF score
    val targetVector = tfidfDF.filter(col("user_id") === targetUser)
      .select("word", "tf_idf")
      .withColumnRenamed("tf_idf", "target_tfidf")

    // Join the TF-IDF dataset with the target user's vector based on the "word" column
    // Exclude the target user's data and compute the dot product of TF-IDF values
    val joinedDF = tfidfDF
      .join(targetVector, "word")
      .filter(col("user_id") =!= targetUser)
      .withColumn("dot_product", col("tf_idf") * col("target_tfidf"))

    // Aggregate the dot product for each user to compute the numerator of the cosine similarity formula
    val numerator = joinedDF.groupBy("user_id")
      .agg(sum("dot_product").alias("numerator"))

    // Compute the squared TF-IDF values for each user to calculate the norm
    val normDF = tfidfDF.withColumn("squared_tfidf", col("tf_idf") * col("tf_idf"))

    // Aggregate the squared values and take the square root to compute the norm for each user
    val userNorms = normDF.groupBy("user_id")
      .agg(sqrt(sum("squared_tfidf")).alias("user_norm"))

    // Compute the norm for the target user by filtering their data
    val targetNorm = normDF.filter(col("user_id") === targetUser)
      .select(sqrt(sum("squared_tfidf")).alias("target_norm"))
      .as[Double]
      .collect()
      .head // Extract the norm value as a scalar

    // Join the numerator with user norms and calculate the cosine similarity
    // Cosine similarity = numerator / (norm of target user * norm of each user)
    val similarityDF = numerator.join(userNorms, "user_id")
      .withColumn("cosine_similarity", col("numerator") / (col("user_norm") * lit(targetNorm)))

    // Order the users by similarity in descending order, and take the top 3 IDs in a list
    similarityDF.orderBy(desc("cosine_similarity"))
      .limit(3)
      .select("user_id")
      .as[Int]
      .collect()
      .toList
  }

  /*
   * 8971360
   * 11277999
   * 9911449
   */


  /**
   * Generates and prints final game recommendations for a target user based on games played by similar users
   *
   * @param top3Users   List of IDs of the most similar users.
   * @param targetUser  ID of the target user.
   * @param gamesTitles DataFrame containing game titles.
   * @param userGamesData  Complete, cleaned, and merged dataset.
   */
  private def generateFinalRecommendations(top3Users: List[Int], targetUser: Int, gamesTitles: DataFrame, userGamesData: DataFrame): Unit = {
    val gamesByTopUsers = userGamesData.filter(col("user_id").isin(top3Users: _*)).select("app_id", "user_id")
    val gamesByTargetUser = userGamesData.filter(col("user_id") === targetUser).select("app_id")

    val recommendedGames = gamesByTopUsers.join(gamesByTargetUser, Seq("app_id"), "left_anti")
    val finalRecommendations = recommendedGames.join(gamesTitles.select("app_id", "title"), Seq("app_id"))
      .groupBy("app_id","title")
      .agg(collect_list("user_id").alias("user_ids"))

    finalRecommendations.show(finalRecommendations.count().toInt, truncate = false)
  }
}
