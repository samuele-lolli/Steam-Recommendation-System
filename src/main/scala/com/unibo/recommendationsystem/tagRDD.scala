package com.unibo.recommendationsystem

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession.builder
import org.apache.spark.sql.types.{ArrayType, BooleanType, DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.storage.StorageLevel

import scala.reflect.ClassTag

object tagRDD {
  def main(args: Array[String]): Unit = {

    //Initialize SparkSession
    val spark = builder
      .appName("recommendationsystem")
      .config("spark.master", "local[*]")
      .getOrCreate()


    val dataPathRec = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/recommendations.csv"
    val dataPathGames = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/games.csv"
    val metadataPath = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/games_metadata.json"

    val recSchema = StructType(Array(
      StructField("app_id", IntegerType, nullable = false), // ID del gioco
      StructField("helpful", IntegerType, nullable = true), // Numero di voti utili
      StructField("funny", IntegerType, nullable = true), // Numero di voti divertenti
      StructField("date", StringType, nullable = true), // Data della recensione
      StructField("is_recommended", BooleanType, nullable = true), // Recensione positiva o negativa
      StructField("hours", DoubleType, nullable = true), // Ore di gioco
      StructField("user_id", IntegerType, nullable = false), // ID utente
      StructField("review_id", IntegerType, nullable = false) // ID recensione
    ))

    val gamesSchema = StructType(Array(
      StructField("app_id", IntegerType, nullable = false), // ID del gioco
      StructField("title", StringType, nullable = true), // Titolo del gioco
      StructField("date_release", StringType, nullable = true), // Data di rilascio
      StructField("win", BooleanType, nullable = true), // Disponibile per Windows
      StructField("mac", BooleanType, nullable = true), // Disponibile per Mac
      StructField("linux", BooleanType, nullable = true), // Disponibile per Linux
      StructField("rating", StringType, nullable = true), // Valutazione del gioco
      StructField("positive_ratio", IntegerType, nullable = true), // Percentuale di recensioni positive
      StructField("user_reviews", IntegerType, nullable = true), // Numero di recensioni utente
      StructField("price_final", DoubleType, nullable = true), // Prezzo finale
      StructField("price_original", DoubleType, nullable = true), // Prezzo originale
      StructField("discount", DoubleType, nullable = true), // Sconto
      StructField("steam_deck", BooleanType, nullable = true) // CompatibilitÃ con Steam Deck
    ))

    val metadataSchema = StructType(Array(
      StructField("app_id", IntegerType, nullable = false),
      StructField("description", StringType, nullable = true),
      StructField("tags", ArrayType(StringType), nullable = true) // Array di stringhe per i tag
    ))

    //PREPROCESSING
    val tPreProcessingI = System.nanoTime()

    val dfRec = spark.read.format("csv").option("header", "true").schema(recSchema).load(dataPathRec).filter("is_recommended = true")
    val dfGames = spark.read.format("csv").option("header", "true").schema(gamesSchema).load(dataPathGames)
    val dfMetadata = spark.read.format("json").schema(metadataSchema).load(metadataPath)

    // Step 1: Use Higher-order function to process the recommendation RDD
    def processRDD[T: ClassTag](rdd: RDD[org.apache.spark.sql.Row], processingFunc: org.apache.spark.sql.Row => T): RDD[T] = {
      rdd.map(processingFunc)
    }

    // Change the structure to include userId
    val selectedRecRDD: RDD[(Int, String)] = processRDD(dfRec.rdd, row => (row.getInt(0), row.getInt(6).toString))

    // Step 2: Create a higher-order function to map game titles
    def mapGames[T](rdd: RDD[org.apache.spark.sql.Row], mappingFunc: org.apache.spark.sql.Row => (Int, T)): Map[Int, T] = {
      rdd.map(mappingFunc).collect().toMap
    }

    // Create title dictionary mapping appId to title
    val tags = mapGames(dfMetadata.rdd, row => (row.getInt(0), row.getList(2).toArray.map(_.toString).mkString(",").toLowerCase.trim.replaceAll("\\s+", " ")))

    // Broadcast the tagMap to all workers
    val broadcastTagMap = spark.sparkContext.broadcast(tags)

    // Update mergeRDDs to no longer require tagMap as a parameter
    def mergeRDDs[T: ClassTag](recRDD: RDD[(Int, String)], mapFunc: (Int, String) => T): RDD[T] = {
      recRDD.map { case (appId, userId) => mapFunc(appId, userId) }
    }

    // Update mapFunc to use broadcastTagMap instead of passing tagMap
    val mergedRDD = mergeRDDs(selectedRecRDD, (appId: Int, userId: String) => {
      val tag = broadcastTagMap.value.getOrElse(appId, "")  // Retrieve the game title
      (appId, tag, userId)  // Now include userId in the output
    })
      .filter { case (_, tag, _) => tag.nonEmpty }  // Filter out empty tags
      .persist(StorageLevel.MEMORY_AND_DISK)


    // Step 4: Aggregate data by user using a higher-order function to pass a filtering condition
    def aggregateByUser[T](rdd: RDD[(String, Array[String])], filterFunc: Array[String] => Boolean): RDD[(String, Array[String])] = {
      rdd.filter { case (_, words) => filterFunc(words) }
    }

    // Define filtering logic as a function
    val minWords = 0
    val filterCondition = (words: Array[String]) => words.length >= minWords

    val aggregateDataRDD = mergedRDD
      .map { case (_, title, user) => (user, title.split("\\s+")) } // Split title into words
      .reduceByKey { (arr1, arr2) => arr1 ++ arr2 } // Concatenate arrays of words

    val filteredAggregateDataRDD = aggregateByUser(aggregateDataRDD, filterCondition)

    def explodeRDD[T: ClassTag](rdd: RDD[(String, Array[String])], explodeFunc: (String, Array[String]) => Iterable[T]): RDD[T] = {
      rdd.flatMap { case (userId, words) => explodeFunc(userId, words) }
    }

    val explodedRDD = explodeRDD(filteredAggregateDataRDD, (userId, words) => words.map(word => (userId, word)))

    val tPreProcessingF = System.nanoTime()

    val tTFIDFI = System.nanoTime()

    // Step 6: TF-IDF function definition using higher-order functions
    def calculateTFIDF[T](userWordsDataset: RDD[(String, String)],
                          tfFunc: String => Map[String, Double],
                          idfFunc: RDD[(String, String)] => Map[String, Double]): RDD[(String, Map[String, Double])] = {

      // Group words by user
      val groupedUserWords = userWordsDataset
        .map { case (userId, words) => (userId, words) } // Keep as a (userId, words) pair
        .reduceByKey { (words1, words2) => words1 + "," + words2 } // Concatenate words with a comma
        .persist(StorageLevel.MEMORY_AND_DISK)

      // Use the provided IDF function
      val idfValues = idfFunc(groupedUserWords)

      // Calculate TF-IDF
      groupedUserWords.map { case (user, words) =>
        val tfValues = tfFunc(words)
        val tfidfValues = tfValues.map { case (word, tf) => (word, tf * idfValues.getOrElse(word, 0.0)) }
        (user, tfidfValues)
      }
    }

    // Define term frequency (TF) calculation logic
    val calculateTF = (userWords: String) => {
      val wordsSplit = userWords.split(",")
      val totalWords = wordsSplit.size.toDouble
      wordsSplit.groupBy(identity).mapValues(_.length / totalWords)
    }

    // Define inverse document frequency (IDF) calculation logic
    val calculateIDF = (userWords: RDD[(String, String)]) => {
      val userCount = userWords.count()  // Total number of users (or documents)

      // Directly compute IDF without storing intermediate 'wordsCount'
      userWords
        .flatMap { case (_, words) => words.split(",").distinct }  // Split and get distinct words
        .map(word => (word, 1))  // Map each distinct word to (word, 1)
        .reduceByKey(_ + _)      // Reduce by key to get the count of each word
        .map { case (word, count) => (word, math.log(userCount.toDouble / count)) }  // Compute IDF
        .collect()
        .toMap
    }

    val tfidfValues = calculateTFIDF(explodedRDD, calculateTF, calculateIDF)

    val tTFIDFF = System.nanoTime()

    val tCosineSimilarityI = System.nanoTime()


    // Step 7: Higher-order function to compute cosine similarity
    def computeCosineSimilarity(vector1: Map[String, Double], vector2: Map[String, Double], dotProductFunc: (Map[String, Double], Map[String, Double]) => Double, magnitudeFunc: Map[String, Double] => Double): Double = {
      dotProductFunc(vector1, vector2) / (magnitudeFunc(vector1) * magnitudeFunc(vector2))
    }

    // Define dot product and magnitude logic
    val dotProduct = (v1: Map[String, Double], v2: Map[String, Double]) => {
      v1.foldLeft(0.0) { case (acc, (key, value)) =>
        acc + v2.getOrElse(key, 0.0) * value
      }
    }

    val magnitude = (vector: Map[String, Double]) => {
      math.sqrt(vector.values.map(value => value * value).sum)
    }

    // Step 8: Higher-order function to get similar users

    def getSimilarUsers[T: ClassTag](
                                      targetUser: Int,
                                      tfidfValues: RDD[(String, Map[String, Double])],
                                      similarityFunc: (Map[String, Double], Map[String, Double]) => Double
                                    ): Array[(String, Double)] = {

      // Filter to get the target user's games

      val targetUserGamesRDD = tfidfValues
        .filter { case (userId, _) => userId == targetUser.toString }
        .map { case (_, gameVector) => gameVector }
      //println(s"Time for filtering: ${(tFilterEnd - tFilterStart) / 1e9} seconds")

      // Step 2: Check if the target user has any games
      if (!targetUserGamesRDD.isEmpty()) {
        // Step 3: Collect the target user's games (should be a single map)
        val targetUserGames = targetUserGamesRDD.collect().head
        // println(s"Time for collecting: ${(tCollectEnd - tCollectStart) / 1e9} seconds")


        // Step 3: Compute similarity for other users
        val similarUsers = tfidfValues
          .filter(_._1 != targetUser.toString) // Filter out the target user
          .map { case (otherUserId, otherUserGames) =>
            (otherUserId, similarityFunc(targetUserGames, otherUserGames))
          }
          .collect() // Collect the results to the driver
          .sortBy(-_._2) // Sort by similarity (descending)
          .take(3) // Take top 10 similar users

        similarUsers // Return the list of similar users
      } else {
        // Step 4: Return an empty array if the target user has no games
        Array.empty[(String, Double)]
      }
    }

    // Use cosine similarity logic
    val cosineSimilarity = (v1: Map[String, Double], v2: Map[String, Double]) =>
      computeCosineSimilarity(v1, v2, dotProduct, magnitude)

    val targetUser = 4893896
    val recommendations = getSimilarUsers(targetUser, tfidfValues, cosineSimilarity)

    val tCosineSimilarityF = System.nanoTime()

    // Get final recommendations
    val tFinalRecommendI = System.nanoTime()

    // Extract appIds played by target user
    val appIdsPlayedByTargetUser = mergedRDD
      .filter { case (_, _, user) => user == targetUser.toString }  // Filter by targetUser
      .map(_._1)  // Extract appId
      .distinct()  // Get unique appIds
      .collect()   // Collect appIds into an array
      .toSet       // Convert to a Set for easy lookup

    // Convert dfGames to an RDD of (appId, title)
    val dfGamesRDD = dfGames.rdd.map(row => (row.getAs[Int]("app_id"), row.getAs[String]("title")))

    // Filter dfGamesRDD by appId and extract titles
    val titlesPlayedByTargetUser = dfGamesRDD
      .filter { case (appId, _) => appIdsPlayedByTargetUser.contains(appId) }  // Filter by appId in the set
      .map(_._2)  // Extract the titles
      .distinct() // Ensure unique titles
      .collect()  // Collect the titles into an array
      .toSet      // Convert to a Set of titles


    // Use higher-order function for filtering and mapping final recommendations
    def filterAndMap[T: ClassTag](rdd: RDD[(Int, String, String)],
                                  filterFunc: ((Int, String, String)) => Boolean,
                                  mapFunc: ((Int, String, String)) => T): RDD[T] = {
      rdd.filter(filterFunc).map(mapFunc)
    }

    val userIdsToFind = recommendations.take(3).map(_._1).toSet

    println("Top 3 similar users")
    recommendations.take(3).foreach(println)

    val finalRecommendations = filterAndMap(mergedRDD,
      { case (_, tag, user) => userIdsToFind.contains(user) && !titlesPlayedByTargetUser.contains(tag)},
      { case (appId, tag, user) => (appId, tag, user) })

    // Step 2: Prepare finalRecommendations by mapping the appId to the title
    val finalRecommendationsWithTitle = finalRecommendations
      .map { case (appId, tags, userId) => (appId, (tags, userId)) }  // Prepare for join
      .join(dfGamesRDD)  // Join with the RDD containing (appId, title)
      .map { case (appId, ((_, userId), title)) => (appId, title, userId) } // Replace tags with title

    val groupedRecommendations = finalRecommendationsWithTitle
      .map { case (appId, title, userId) => (appId, (title, userId)) }  // Prepare for grouping
      .reduceByKey { case ((title1, userId1), (_, userId2)) =>
        // Assuming title is the same for all records with the same appId
        val title = title1 // or title2, both are the same
        val userIds = Set(userId1, userId2).mkString(",") // Use a Set to avoid duplicates
        (title, userIds)  // This maintains userIds as a Set
      }
      .map { case (appId, (title, userIds)) =>
        (appId, title, userIds)  // Return as (appId, title, comma-separated userIds)
      }

    groupedRecommendations.collect().foreach { case (_, title, userIds) =>
      println(s"userId: $userIds, title: $title")
    }

    val tFinalRecommendF = System.nanoTime()


    println(s"\n\nExecution time(preprocessing): ${(tPreProcessingF - tPreProcessingI) / 1000000}ms\n")
    println(s"\n\nExecution time(Tf-Idf calculation): ${(tTFIDFF - tTFIDFI) / 1000000}ms\n")
    println(s"\n\nExecution time(Cosine similarity calculation): ${(tCosineSimilarityF - tCosineSimilarityI) / 1000000}ms\n")
    println(s"\n\nExecution time(final recommendation): ${(tFinalRecommendF - tFinalRecommendI) / 1000000}ms\n")
    println(s"\n\nExecution time(total): ${(tFinalRecommendF - tPreProcessingI) / 1000000}ms\n")


    /*
    (1299120,Mosaique Neko Waifus 2,13498880,10381354,11346100)
(1211360,NEOMORPH,13498880)
(382560,Hot Lava,13498880)
(333600,NEKOPARA Vol. 1,11346100,13498880)
(748480,Wild Romance: Mofu Mofu Edition,11346100,10381354)
(1222160,Chinese Brush Simulator,13498880)
(48000,LIMBO,10381354)
(240720,Getting Over It with Bennett Foddy,13498880)
(356400,Thumper,13498880)
(435120,Rusty Lake Hotel,10381354)
(687920,Tropical Liquor,10381354)
(1194560,单身狗的最后机会,10381354)
(1607280,YUME 2 : Sleepless Night,10381354)
(1377360,Vampires' Melody,10381354)
(1570960,Grey Instinct,11346100)
(1186400,West Sweety,10381354,13498880)
(1418160,Happy Quest,10381354,11346100,13498880)
(1252560,Love Breakout,13498880,10381354)
(6000,STAR WARS™ Republic Commando™,11346100)
(1012880,60 Seconds! Reatomized,13498880)
(32370,STAR WARS™ - Knights of the Old Republic™,11346100)
(1393410,Seek Girl V,10381354,13498880)
(1058530,H-Rescue,10381354,13498880)
(1419730,Seek Girl Ⅵ,13498880,10381354,11346100)
(410850,DRAGON QUEST HEROES™ Slime Edition,11346100)
(899970,NEKOPARA Extra,11346100)
(1464930,Seek Girl Ⅶ,11346100,13498880,10381354)
(1710930,Bubble People,10381354)
(1256610,Dream Date,13498880,10381354)
(692850,Bloodstained: Ritual of the Night,11346100)
(540610,Delicious! Pretty Girls Mahjong Solitaire,11346100)
(407330,Sakura Dungeon,11346100)
(862690,Kidnapper: Gosh I'm Kidnapped by a Pupil,11346100)
(1722770,Dream Girls Collection,11346100)
(569810,LEAVES - The Return,13498880)
(17410,Mirror's Edge™,11346100)
(844930,Fox Hime Zero,10381354)
(1102130,Florence,10381354)
(822930,Wolf Tails,11346100)
(1109570,Word Game,10381354)
(1422610,Aurora,11346100,13498880)
(1576130,IdolDays,10381354)
(803330,Destroy All Humans!,11346100)
(1605010,Adorable Witch,11346100,10381354)
(998930,Seek Girl,11346100)
(1222690,Dragon Age™ Inquisition,11346100)
(47810,Dragon Age: Origins - Ultimate Edition,11346100)
(1126290,Lost,13498880,10381354)
(1385730,Mosaique Neko Waifus 3,10381354,11346100,13498880)
(501300,What Remains of Edith Finch,10381354)
(35140,Batman: Arkham Asylum Game of the Year Edition,11346100)
(1067540,Röki,10381354)
(1611300,Happy Guy,10381354)
(1504020,Mosaique Neko Waifus 4,10381354,13498880,11346100)
(1238020,Mass Effect™ 3 N7 Digital Deluxe Edition (2012),11346100)
(200260,Batman: Arkham City - Game of the Year Edition,11346100)
(1548820,Happy Puzzle,13498880,10381354,11346100)
(17460,Mass Effect (2007),11346100)
(356500,STAR WARS™ Galactic Battlegrounds Saga,11346100)
(1732740,Sakura Hime,10381354)
(1709460,Hot And Lovely 4,13498880,10381354)
(6020,STAR WARS™ Jedi Knight - Jedi Academy™,11346100)
(384180,Prominence Poker,13498880)
(939620,Pleasure Puzzle:Portrait 趣拼拼：肖像画,10381354)
(1072420,DRAGON QUEST BUILDERS™ 2,11346100)
(1202900,Assemble with Care,10381354)
(208580,STAR WARS™ Knights of the Old Republic™ II - The Sith Lords™,11346100)
(654820,Akin Vol 2,13498880)
(1038740,Fluffy Store,10381354)
(972660,Spiritfarer®: Farewell Edition,10381354)
(1306580,Left on Read,11346100)
(828070,Treasure Hunter Claire,11346100)
(22230,Rock of Ages,11346100)
(368230,Kingdom: Classic,10381354)
(487430,KARAKARA,11346100)
(1502230,Tower of Waifus,10381354,13498880)
(1618230,恋爱关系/Romance,13498880)
(1354230,Love Tavern,11346100,10381354)
(802870,The Ditzy Demons Are in Love With Me,10381354,13498880)
(1101270,Anime Artist,11346100)
(1135830,Brave Alchemist Colette,11346100)
(32470,STAR WARS™ Empire at War - Gold Pack,11346100)
(1393350,Swaying Girl,13498880)
(1008710,Wet Girl,13498880,11346100)
(577670,Demolish & Build 2018,13498880)
(969990,SpongeBob SquarePants: Battle for Bikini Bottom - Rehydrated,11346100)
(914710,Cat Quest II,13498880)
(1153430,Love wish,10381354,13498880)
(1295510,DRAGON QUEST® XI S: Echoes of an Elusive Age™ - Definitive Edition,11346100)
(1146630,Yokai's Secret,10381354,11346100)
(1986230,What if your girl was a frog?,11346100)
(1908870,'LIFE' not found;,11346100)
(1336790,Mini Words: Top Games,11346100)
(1150950,Timelie,10381354)
(1508680,Love n War: Warlord by Chance,10381354)
(1460040,Love Fantasy,13498880)
(1113560,NieR Replicant™ ver.1.22474487139...,11346100)
(1127400,Mindustry,13498880)
(32440,LEGO® Star Wars™ - The Complete Saga,11346100)
(1523400,Livestream: Escape from Hotel Izanami,10381354)


Execution time(preprocessing): 10299ms



Execution time(Tf-Idf calculation): 77646ms



Execution time(Cosine similarity calculation): 103851ms



Execution time(final recommendation): 883ms



Execution time(total): 192681ms
     */
    spark.stop()
  }
}