package com.unibo.recommendationsystem

import com.unibo.recommendationsystem.utils.{schemaUtils, timeUtils}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Encoder, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.collection.Map

class sqlRecommendation (spark: SparkSession, dataPathRec: String, dataPathGames: String, metadataPath: String) {


  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    //Elapsed time for Preprocessing Data:	1495ms (1495704000ns)
    val (explodedDF, filteredData, gamesTitles, cleanMerge) = timeUtils.time(preprocessData(), "Preprocessing Data", "SQL")
    println("Calculate term frequency and inverse document frequency...")
    val tfidfValues = timeUtils.time(calculateTFIDF(explodedDF, filteredData), "Calculating TF-IDF", "SQL")
    println("Calculate cosine similarity to get similar users...")
    val topUsersSimilarity = timeUtils.time(computeCosineSimilarity(tfidfValues, targetUser), "Getting Similar Users", "SQL")
    println("Calculate final recommendation...")
    timeUtils.time(getFinalRecommendations(topUsersSimilarity, targetUser, gamesTitles, cleanMerge), "Generating Recommendations", "SQL")
  }

  private def preprocessData(): (DataFrame, DataFrame, DataFrame, DataFrame) = {

    val dfRec = spark.read.format("csv").option("header", "true").schema(schemaUtils.recSchema).load(dataPathRec).filter("is_recommended = true")
    val dfGames = spark.read.format("csv").option("header", "true").schema(schemaUtils.gamesSchema).load(dataPathGames)
    val dfMetadata = spark.read.format("json").schema(schemaUtils.metadataSchema).load(metadataPath)

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

    val gamesTitles = dfGames.select("app_id", "title")

    (explodedDF, filteredData, gamesTitles, cleanMerge)
  }

  private def calculateTFIDF(explodedDF: DataFrame, filteredData: DataFrame): DataFrame = {
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

    tfidfDF
  }

  def computeCosineSimilarity(tfidfDF: DataFrame, targetUser: Int): List[Int] = {
    // Define the cosine similarity function
    def calculateCosineSimilarity(vector1: Map[String, Double], vector2: Map[String, Double], dotProductFunc: (Map[String, Double], Map[String, Double]) => Double, magnitudeFunc: Map[String, Double] => Double): Double = {
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
      val cosineSimilarity = calculateCosineSimilarity(targetUserVector, userVector, dotProduct, magnitude)
      (userId, cosineSimilarity)
    }.toDF("user_id", "cosine_similarity")


    // Step 3: Get the top 3 users with highest cosine similarity
    val top3Users = otherUsersWithSimilarity.orderBy(desc("cosine_similarity")).limit(3)

    val topSimilarUsers = top3Users.select("user_id").collect().map(row => row.getAs[Int]("user_id")).toList

    topSimilarUsers
  }

  /*

  Top 3 users with highest cosine similarity:
userId: 8971360, cosine similarity: 0.8591424719530733
userId: 11277999, cosine similarity: 0.8436706750570966
userId: 9911449, cosine similarity: 0.8421752054744202
Elapsed time for Getting Similar Users:	533863ms (533863187600ns)
  */

  def getFinalRecommendations(top3Users: List[Int], targetUser: Int, gamesTitles: DataFrame, cleanMerge: DataFrame) = {

    val gamesByTopUsers = cleanMerge.filter(col("user_id").isin(top3Users: _*))  // Use : _* to expand the list
      .select("app_id", "user_id")

    // Step 3: Fetch the games played by the target user
    val gamesByTargetUser = cleanMerge.filter(col("user_id") === targetUser)
      .select("app_id")

    // Step 4: Exclude the games played by the target user from the games played by the similar users
    val recommendedGames = gamesByTopUsers.join(gamesByTargetUser, Seq("app_id"), "left_anti")

    // Step 5: Join with dfGames to get the titles of the recommended games
    val finalRecommendations = recommendedGames
      .join(gamesTitles.select("app_id", "title"), Seq("app_id"))
      .select("title", "user_id")

    // Show the resulting DataFrame with titles and users
    val groupedRecommendations = finalRecommendations
      .groupBy("title")
      .agg(collect_list("user_id").alias("user_ids")) // Aggregate user_ids for each title
      .select("title", "user_ids") // Select only the title and aggregated user_ids

    groupedRecommendations.show(groupedRecommendations.count.toInt, truncate = false)
  }
  /*
  Elapsed time for Generating Recommendations:	384205ms (384205182800ns)
  +---------------------------------------------------------------------------------+------------------+
  |title                                                                            |user_ids          |
  +---------------------------------------------------------------------------------+------------------+
  |Wallpaper Engine                                                                 |[8971360]         |
  |Valiant Hearts: The Great War™ / Soldats Inconnus : Mémoires de la Grande Guerre™|[9911449]         |
  |The Operational Art of War IV                                                    |[9911449]         |
  |Mafia III: Definitive Edition                                                    |[9911449]         |
  |Toy Soldiers                                                                     |[9911449]         |
  |Lock 'n Load Tactical Digital: Core Game                                         |[9911449]         |
  |rFactor 2                                                                        |[9911449]         |
  |Far Cry® 5                                                                       |[9911449]         |
  |DCS World Steam Edition                                                          |[9911449]         |
  |Moonbase Alpha                                                                   |[9911449]         |
  |Battle Academy                                                                   |[9911449]         |
  |Making History: The Calm & the Storm                                             |[9911449]         |
  |Super Meat Boy                                                                   |[9911449]         |
  |Jagged Alliance - Back in Action                                                 |[9911449]         |
  |Company of Heroes: Tales of Valor                                                |[9911449]         |
  |Plants vs. Zombies GOTY Edition                                                  |[9911449]         |
  |Alice: Madness Returns                                                           |[9911449]         |
  |Jagged Alliance Flashback                                                        |[9911449]         |
  |Kane and Lynch: Dead Men™                                                        |[9911449]         |
  |FEZ                                                                              |[9911449]         |
  |Thirty Flights of Loving                                                         |[9911449]         |
  |Portal 2                                                                         |[9911449]         |
  |Total War: ROME II - Emperor Edition                                             |[9911449]         |
  |RaceRoom Racing Experience                                                       |[9911449]         |
  |American Truck Simulator                                                         |[11277999]        |
  |Command: Modern Operations                                                       |[9911449]         |
  |Just Cause 2: Multiplayer Mod                                                    |[11277999]        |
  |Assetto Corsa Competizione                                                       |[9911449]         |
  |GASP                                                                             |[9911449]         |
  |The Few                                                                          |[9911449]         |
  |Jagged Alliance 2 - Wildfire                                                     |[9911449]         |
  |Flight Control HD                                                                |[9911449]         |
  |Sailaway - The Sailing Simulator                                                 |[9911449]         |
  |Battlezone 98 Redux                                                              |[9911449]         |
  |Swords and Soldiers HD                                                           |[9911449]         |
  |Knights of Honor                                                                 |[9911449]         |
  |Deadlight                                                                        |[9911449]         |
  |Arma 3                                                                           |[9911449]         |
  |Baldur's Gate: Enhanced Edition                                                  |[9911449]         |
  |Company of Heroes                                                                |[11277999]        |
  |Grand Theft Auto V                                                               |[9911449]         |
  |Close Combat - Gateway to Caen                                                   |[9911449]         |
  |Total War: SHOGUN 2                                                              |[8971360]         |
  |Zeno Clash                                                                       |[9911449]         |
  |Lead and Gold: Gangs of the Wild West                                            |[9911449]         |
  |Carrier Deck                                                                     |[9911449]         |
  |ENDLESS™ Space - Definitive Edition                                              |[9911449]         |
  |The Walking Dead                                                                 |[9911449]         |
  |Chivalry: Medieval Warfare                                                       |[9911449]         |
  |Dungeon Siege III                                                                |[9911449]         |
  |Mount & Blade: With Fire & Sword                                                 |[9911449]         |
  |Guacamelee! Gold Edition                                                         |[9911449]         |
  |Assetto Corsa                                                                    |[9911449]         |
  |Dragon's Dogma: Dark Arisen                                                      |[9911449]         |
  |Flashpoint Campaigns: Red Storm Player's Edition                                 |[9911449]         |
  |Gunpoint                                                                         |[9911449]         |
  |Call of Duty®: Modern Warfare® 2 (2009)                                          |[9911449]         |
  |Train Simulator Classic                                                          |[9911449]         |
  |Darkest Dungeon®                                                                 |[9911449]         |
  |Unity of Command: Stalingrad Campaign                                            |[9911449]         |
  |Planet Coaster                                                                   |[9911449]         |
  |Day of Defeat: Source                                                            |[9911449]         |
  |IL-2 Sturmovik: 1946                                                             |[9911449]         |
  |romantic player                                                                  |[9911449]         |
  |Motorsport Manager                                                               |[9911449]         |
  |Pride of Nations                                                                 |[9911449]         |
  |Worms Reloaded                                                                   |[9911449]         |
  |F.E.A.R.                                                                         |[9911449]         |
  |Hitman 2: Silent Assassin                                                        |[9911449]         |
  |Wings of Prey                                                                    |[9911449]         |
  |Military Life: Tank Simulator                                                    |[9911449]         |
  |Ultimate General: Gettysburg                                                     |[9911449]         |
  |Sébastien Loeb Rally EVO                                                         |[9911449]         |
  |Crusader Kings II                                                                |[9911449]         |
  |No Man's Sky                                                                     |[9911449]         |
  |X-Plane 11                                                                       |[9911449]         |
  |Red Orchestra 2: Heroes of Stalingrad with Rising Storm                          |[9911449, 8971360]|
  |Europa Universalis IV                                                            |[9911449]         |
  |IL-2 Sturmovik: Cliffs of Dover                                                  |[9911449]         |
  |Shadowrun Returns                                                                |[9911449]         |
  |Tank On Tank Digital  - West Front                                               |[9911449]         |
  |Velvet Assassin                                                                  |[9911449]         |
  |INSURGENCY: Modern Infantry Combat                                               |[9911449]         |
  |Command Ops 2 Core Game                                                          |[9911449]         |
  |Black Mesa                                                                       |[9911449]         |
  |XCOM: Enemy Unknown                                                              |[9911449]         |
  |Simutrans                                                                        |[9911449]         |
  |Euro Truck Simulator 2                                                           |[9911449]         |
  |ABZU                                                                             |[9911449]         |
  |Valentino Rossi The Game                                                         |[9911449]         |
  |Rise of Nations: Extended Edition                                                |[9911449]         |
  |Knights and Merchants                                                            |[9911449]         |
  |articy:draft 3                                                                   |[9911449]         |
  |Gary Grigsby's War in the East                                                   |[9911449]         |
  |MORDHAU                                                                          |[9911449]         |
  |Theatre of War                                                                   |[9911449]         |
  |Unstoppable Gorg                                                                 |[9911449]         |
  |Worms Crazy Golf                                                                 |[9911449]         |
  |Sid Meier's Railroads!                                                           |[9911449]         |
  |Victoria 3                                                                       |[9911449]         |
  |NASCAR '15 Victory Edition                                                       |[9911449]         |
  |Unturned                                                                         |[8971360]         |
  |Combat Mission Shock Force 2                                                     |[9911449]         |
  |Rise of Flight United                                                            |[9911449]         |
  |Panzer Tactics HD                                                                |[9911449]         |
  |Advanced Tactics Gold                                                            |[9911449]         |
  |The Tiny Bang Story                                                              |[9911449]         |
  |Homeworld Remastered Collection                                                  |[9911449]         |
  |Shop Heroes                                                                      |[9911449]         |
  |The Expendabros                                                                  |[9911449]         |
  |WARTILE                                                                          |[9911449]         |
  |Rise of Prussia Gold                                                             |[9911449]         |
  |Total War: MEDIEVAL II – Definitive Edition                                      |[8971360]         |
  |South Park™: The Stick of Truth™                                                 |[11277999]        |
  |Microsoft Flight Simulator X: Steam Edition                                      |[9911449]         |
  |BRAIN / OUT                                                                      |[9911449]         |
  |Costume Quest                                                                    |[9911449]         |
  |Overcooked                                                                       |[9911449]         |
  |Battle Academy 2: Eastern Front                                                  |[9911449]         |
  |To End All Wars                                                                  |[9911449]         |
  |Tom Clancy's Rainbow Six® Siege                                                  |[9911449]         |
  +---------------------------------------------------
   */
}
