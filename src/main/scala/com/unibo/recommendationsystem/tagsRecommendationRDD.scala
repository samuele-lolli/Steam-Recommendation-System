package com.unibo.recommendationsystem


import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession.builder
import org.apache.spark.sql.types.{ArrayType, BooleanType, DoubleType, IntegerType, StringType, StructField, StructType}

import scala.collection.convert.ImplicitConversions.`collection AsScalaIterable`
import scala.reflect.ClassTag

object tagsRecommendationRDD {
  def main(args: Array[String]): Unit = {

    //Initialize SparkSession
    val spark = builder
      .appName("recommendationsystem")
      .config("spark.master", "local[*]")
      .getOrCreate()

    val dataPathRec = "C:\\Users\\gbeks\\IdeaProjects\\recommendationsystem\\steam-datasets\\recommendations.csv"
    val dataPathGames = "C:\\Users\\gbeks\\IdeaProjects\\recommendationsystem\\steam-datasets\\games.csv"
    val metadataPath = "C:\\Users\\gbeks\\IdeaProjects\\recommendationsystem\\steam-datasets\\games_metadata.json"

    val recSchema = StructType(Array(
      StructField("app_id", IntegerType, nullable = false),
      StructField("helpful", IntegerType, nullable = true),
      StructField("funny", IntegerType, nullable = true),
      StructField("date", StringType, nullable = true),
      StructField("is_recommended", BooleanType, nullable = true),
      StructField("hours", DoubleType, nullable = true),
      StructField("user_id", IntegerType, nullable = false),
      StructField("review_id", IntegerType, nullable = false)
    ))

    val gamesSchema = StructType(Array(
      StructField("app_id", IntegerType, nullable = false),
      StructField("title", StringType, nullable = true),
      StructField("date_release", StringType, nullable = true),
      StructField("win", BooleanType, nullable = true),
      StructField("mac", BooleanType, nullable = true),
      StructField("linux", BooleanType, nullable = true),
      StructField("rating", StringType, nullable = true),
      StructField("positive_ratio", IntegerType, nullable = true),
      StructField("user_reviews", IntegerType, nullable = true),
      StructField("price_final", DoubleType, nullable = true),
      StructField("price_original", DoubleType, nullable = true),
      StructField("discount", DoubleType, nullable = true),
      StructField("steam_deck", BooleanType, nullable = true)
    ))

    val metadataSchema = StructType(Array(
      StructField("app_id", IntegerType, nullable = false),
      StructField("description", StringType, nullable = true),
      StructField("tags", ArrayType(StringType), nullable = true)
    ))

    // PREPROCESSING
    val tPreProcessingI = System.nanoTime()

    val dfRec = spark.read.format("csv").option("header", "true").schema(recSchema).load(dataPathRec).filter("is_recommended = true")
    val dfGames = spark.read.format("csv").option("header", "true").schema(gamesSchema).load(dataPathGames)
    val dfMetadata = spark.read.format("json").schema(metadataSchema).load(metadataPath)

    // RDD for recommendations
    val selectedRecRDD = dfRec.rdd.map(row => (row.getInt(6), row.getInt(0))) // (user_id, app_id)

    // RDD for tags
    val tagsRDD = dfMetadata.rdd.map(row => (row.getInt(0), row.getList(row.fieldIndex("tags")))) // (app_id, List[tags])

    // RDD with one line per recommendation. Result -> (app_id, user_id, List[tags])
    // If game has no tags it goes away as it won't add any infos to our recommendation
    val resultRDD = selectedRecRDD
      .map { case (userId, appId) => (appId, userId) }
      .leftOuterJoin(tagsRDD)
      .map { case (appId, (userId, tagsOpt)) =>
        val tags = tagsOpt.map(_.toList).getOrElse(List.empty[String])
        (appId, userId, tags)
      }  .filter { case (_, _, tags) => tags.nonEmpty }

    // RDD with one line per user. Result -> (user_id, List[app_id], List[tags]
    // The tags of all of a user's games are merged together to create a "user profile"
    val userGamesAndTagsRDD = resultRDD
      .groupBy(_._2)
      .map { case (userId, userGames) =>
        val games = userGames.map(_._1).toList
        val tags = userGames.flatMap(_._3).toList
        (userId, games, tags)
      }.persist()

    // Wrap games and tags into a tuple
    val userGamesAndTagsRDD2: RDD[(Int, (List[Int], List[String]))] = userGamesAndTagsRDD
      .map { case (userId, games, tags) =>
        (userId, (games, tags))
      }

    // Holds user_id and app_ids association to create final recommendations
    val userIdAndGamesRDD = userGamesAndTagsRDD2.mapValues(_._1).persist()

    // Convert tags List to Array
    val filteredAggregateDataRDD = userGamesAndTagsRDD
      .map { case (userId, _, tags) =>
        (userId.toString, tags.toArray)
      }

    // Define function for RDD explosion
    def explodeRDD[T: ClassTag](rdd: RDD[(String, Array[String])], explodeFunc: (String, Array[String]) => Iterable[T]): RDD[T] = {
      rdd.flatMap { case (userId, words) => explodeFunc(userId, words) }
    }

    // Explode RDD with user_ids and tags
    val explodedRDD = explodeRDD(filteredAggregateDataRDD, (userId, words) => words.map(word => (userId, word)))

    val tPreProcessingF = System.nanoTime()
    val tTFIDFI = System.nanoTime()

    // TF-IDF function definition using higher-order functions
    def calculateTFIDF[T](userWordsDataset: RDD[(String, String)],
                          tfFunc: String => Map[String, Double],
                          idfFunc: RDD[(String, String)] => Map[String, Double]): RDD[(String, Map[String, Double])] = {

      // Convert the Array in a single string separated by commas
      val groupedUserWords = userWordsDataset.groupByKey().map { case (userId, wordsIterable) =>
        val wordsString = wordsIterable.mkString(",")
        (userId, wordsString)
      }.persist()

      // Calculate IDF vals
      val idfValues = idfFunc(groupedUserWords)

      // Calculate TF vals and apply TF-IDF
      groupedUserWords.map { case (user, words) =>
        val tfValues = tfFunc(words)
        val tfidfValues = tfValues.map { case (word, tf) => (word, tf * idfValues.getOrElse(word, 0.0)) }
        (user, tfidfValues)
      }
    }

    // Define TF calculation logic
    val calculateTF = (userWords: String) => {
      val wordsSplit = userWords.split(",")
      val totalWords = wordsSplit.size.toDouble
      wordsSplit.groupBy(identity).mapValues(_.length / totalWords)
    }

    // Define IDF calculation logic
    val calculateIDF = (userWords: RDD[(String, String)]) => {
      val userCount = userWords.count()

      // Compute IDF
      userWords
        .flatMap { case (_, words) => words.split(",").distinct }
        .map(word => (word, 1))
        .reduceByKey(_ + _)
        .map { case (word, count) => (word, math.log(userCount.toDouble / count)) }
        .collect()
        .toMap
    }

    val tfidfValues = calculateTFIDF(explodedRDD, calculateTF, calculateIDF)

    val tTFIDFF = System.nanoTime()

    /*
          tfidfvalues.take(10).foreach(println)
          (1445040,Map(Real Time Tactics -> 0.3825598307971532, 2D -> 0.13539112104962048, Indie -> 0.0542217950721404, Choices Matter -> 0.26873002659597295, Simulation -> 0.12652088704721165, Experimental -> 0.3786066388026315, Strategy -> 0.13909851062842016, Space -> 0.2942562010707341, Singleplayer -> 0.07407216588829818, Casual -> 0.09503120826626103, Free to Play -> 0.1999903994062955))
          (1375360,Map(Indie -> 0.5964397457935444))
          (9200,Map(Co-op -> 0.13743410801776018, Online Co-Op -> 0.1679478664117698, Open World -> 0.13241108772872814, Horror -> 0.11776692063896921, FPS -> 0.14077936101074337, RPG -> 0.08077801789922884, Crafting -> 0.1739678927487867, Post-apocalyptic -> 0.18686145060010037, Multiplayer -> 0.1020760026029347, Singleplayer -> 0.040739691238563996, Gore -> 0.13828467530908703, Shooter -> 0.12250224129954071, First-Person -> 0.11049432414995163, Sci-fi -> 0.12577344981324298, Survival -> 0.13705687714746648, Adventure -> 0.04624274550236607, Linear -> 0.1608964380622522, Racing -> 0.15945885598962917, Atmospheric -> 0.08805600151553586, Action -> 0.04219148350050067))
          (1975120,Map(Family Friendly -> 0.15472817355599908, 2D -> 0.09308139572161409, Indie -> 0.037277484112096526, Point & Click -> 0.18145215795460146, Psychological Horror -> 0.17889375275640942, Realistic -> 0.18030761960099573, Exploration -> 0.13599529250472686, Dog -> 0.33307770520555335, Singleplayer -> 0.050924614048204996, Puzzle -> 0.11842601830161197, Casual -> 0.06533395568305446, First-Person -> 0.13811790518743952, Adventure -> 0.057803431877957584, Linear -> 0.20112054757781525, Free to Play -> 0.13749339959182816, Hidden Object -> 0.21648466884650716))
          (967440,Map(Casual -> 0.26133582273221784, Simulation -> 0.34793243937983204, Indie -> 0.1491099364483861, Anime -> 0.5784270956096708))
          (1674960,Map(Indie -> 0.03313554143297469, 3D Platformer -> 0.19942082637473035, Puzzle Platformer -> 0.18756753702581536, Surreal -> 0.20337106651970327, Time Manipulation -> 0.2896379162785132, Colorful -> 0.12155745572513235, Funny -> 0.12908670438231634, Singleplayer -> 0.04526632359840444, Cute -> 0.121579293601025, Runner -> 0.2528427906794611, Logic -> 0.19341691923993873, Casual -> 0.05807462727382618, Precision Platformer -> 0.23313199833724585, Parkour -> 0.2533086680892541, 3D -> 0.11768886833988428, Adventure -> 0.051380828335962296, Platformer -> 0.1407878725109452, Linear -> 0.17877382006916911))
          (556640,Map(Indie -> 0.02982198728967722, Medieval -> 0.17876557792258807, Combat -> 0.15316139375382665, Fighting -> 0.2088760584653398, Dungeon Crawler -> 0.182777549049387, VR -> 0.13992183223768273, Moddable -> 0.2128363052701798, 3D Fighter -> 0.2521349110312074, Singleplayer -> 0.040739691238563996, Gore -> 0.13828467530908703, Violent -> 0.13130266795344483, Sandbox -> 0.1463406427662546, Swordplay -> 0.24499124670790984, First-Person -> 0.11049432414995163, Hack and Slash -> 0.17864769967545535, Adventure -> 0.04624274550236607, Rome -> 0.31284813554933155, Early Access -> 0.11299613265386868, Historical -> 0.17527136989723413, Action -> 0.04219148350050067))
          (6400,Map(Strategy -> 0.38252090422815543, Action -> 0.21095741750250335, RTS -> 0.9279142105727121, Military -> 0.9689819225139589))
          (376240,Map(Indie -> 0.08520567797050634, Text-Based -> 0.5426083476560936, Puzzle -> 0.2706880418322559, Choose Your Own Adventure -> 0.5037406084009808, Adventure -> 0.13212213000676018, Story Rich -> 0.28054715744528513, Interactive Fiction -> 0.5006705701768297))
          (739360,Map(Casual -> 0.3484477636429571, Indie -> 0.19881324859784813, Strategy -> 0.5100278723042072))
    */

    // Target user ID
    val targetUserId = 2591067 //967440 alt

    // Take target TF-IDF vals
    val targetUserTfidf = tfidfValues.filter(_._1.toInt == targetUserId).collect() match {
      case Array(user) => user._2 // Estrai i valori TF-IDF se l'utente è presente
      case _ =>
        println(s"Utente con ID $targetUserId non trovato!")
        Map[String, Double]()
    }

    // Filter target user out of tfidfValues
    val otherUsersTfidf = tfidfValues.filter(_._1.toInt != targetUserId)

    val tCosineSimilarityI = System.nanoTime()

    // Cosine similarity function definition
    def calculateCosineSimilarity(user1Tfidf: Map[String, Double], user2Tfidf: Map[String, Double]): Double = {
      val dotProduct = user1Tfidf.map { case (tag, tfidf) =>
        tfidf * user2Tfidf.getOrElse(tag, 0.0)
      }.sum

      val user1Magnitude = math.sqrt(user1Tfidf.values.map(math.pow(_, 2)).sum)
      val user2Magnitude = math.sqrt(user2Tfidf.values.map(math.pow(_, 2)).sum)

      if (user1Magnitude * user2Magnitude == 0) 0.0 else dotProduct / (user1Magnitude * user2Magnitude)
    }

    // Similarity of target and rest of the RDD
    val userSimilarities = otherUsersTfidf.map { case (userId, userTfidf) =>
      val cosineSimilarity = calculateCosineSimilarity(targetUserTfidf, userTfidf)
      (userId, cosineSimilarity)
    }

    // Similarities outputs
    System.out.println("userSimilarities")
    val sortedSimilarities = userSimilarities.sortBy(_._2, ascending = false)

    println(s"Top 10 most similar users")
    sortedSimilarities.take(10).foreach(println)

    /*
      sortedSimilarities.take(10).foreach(println)
      (194719,0.8257644596959894)
      (8361693,0.8129037586031821)
      (11984700,0.8102376953655179)
      (6928773,0.8000065314747451)
      (441823,0.7992401794172909)
      (6813229,0.7960611178943022)
      (47726,0.7940518653012066)
      (7484104,0.7878510225306956)
      (11191246,0.7874802296803748)
      (11117507,0.7864504783904478)
    */

    // Take 3 most similar users
    val finalUsers = sortedSimilarities.take(3)

    val tCosineSimilarityF = System.nanoTime()
    val tFinalRecommendI = System.nanoTime()

    // Extract IDs
    val finalUserIds = finalUsers.map(_._1.toInt)

    // Extract games from three users with userIdAndGamesRDD and remove duplicates
    val finalUsersGames = userIdAndGamesRDD.filter(x => finalUserIds.contains(x._1))
      .flatMap(_._2)
      .collect().toSet


    /*
      println("Similar users Games IDs:")
      finalUsersGames.foreach(println)
      Similar users Games IDs:
      979690
      1057090
      1369630
      894020
      72000
      330020
      406150
      1147560
      1102190
      283640
      1123050
      1123770
      449960
      600130
    */

    // Take target's games
    val targetUserGames = userIdAndGamesRDD.filter(_._1 == targetUserId).flatMap(_._2).collect().toSet

    /*
      println("Target User Games:")
      targetUserGames.foreach(println)
      Target User Games:
      979690
      1057090
      1369630
      428550
      1147560
      1123770
      266510
    */

    // Remove from recommendations already played games
    val suggestedGames = finalUsersGames.diff(targetUserGames)

    // Print
    // Map games and users who played them
    val gameToUserMap = userIdAndGamesRDD.flatMap { case (userId, games) =>
      games.map(game => (game, userId))
    }.groupByKey().collect().toMap

    // Create a formatted list for output
    val suggestedGamesOutput = suggestedGames.map { game =>
      val usersWhoPlayed = gameToUserMap.getOrElse(game, Iterable.empty).filter(finalUserIds.contains(_)).mkString(", ")
      s"- ${dfGames.filter(s"app_id = $game").select("title").first().getString(0)} (Giocato da: $usersWhoPlayed)"
    }.toList

    // Print games
    println("ID UTENTE TARGET: " + targetUserId)
    println("Utenti simili: " + finalUsers.map(_._1).mkString(", "))
    println("Giochi consigliati:")
    suggestedGamesOutput.foreach(println)

    /*
      ID UTENTE TARGET: 967440
      Utenti simili: 194719, 8361693, 11984700
      Giochi consigliati:
      - Book of Demons (Giocato da: 11984700)
      - Salt and Sanctuary (Giocato da: 194719)
      - Children of Morta (Giocato da: 11984700)
      - Valfaris (Giocato da: 11984700)
      - Closure (Giocato da: 11984700)
      - Death's Door (Giocato da: 11984700)
      - Refunct (Giocato da: 11984700)
      - Monster Train (Giocato da: 11984700)
      - GRIME (Giocato da: 11984700)

      Execution time (preprocessing): 2453 ms
      Execution time (Tf-Idf calculation): 300493 ms
      Execution time (cosine similarity calculation): 232730 ms
      Execution time (final recommendation): 171288 ms
      Execution time (total): 801005 ms
     */

    /*
      ID UTENTE TARGET: 2591067
        Utenti simili: 65064, 13498880, 7002264
        Giochi consigliati:
        - Aurora (Giocato da: 13498880, 65064)
        - Fake Hostel (Giocato da: 7002264)
        - Genital Jousting (Giocato da: 13498880)
        - NALOGI 2 (Giocato da: 7002264)
        - Bunny Girl Story (Giocato da: 65064)
        - King's Bounty II (Giocato da: 65064, 7002264)
        - Love n War: Warlord by Chance (Giocato da: 65064)
        - Adorable Witch (Giocato da: 7002264)
        - NEKO-NIN exHeart 3 (Giocato da: 7002264)
        - SWORD ART ONLINE Alicization Lycoris (Giocato da: 7002264)
        - Hard West 2 (Giocato da: 65064)
        - Startup Company (Giocato da: 7002264)
        - Thumper (Giocato da: 13498880)
        - 60 Seconds! Reatomized (Giocato da: 13498880)
        - Alan Wake (Giocato da: 65064)
        - Demolish & Build 2018 (Giocato da: 13498880)
        - The Ascent (Giocato da: 65064)
        - 9 Monkeys of Shaolin (Giocato da: 65064)
        - Counter-Strike: Source (Giocato da: 7002264)
        - Blackthorn Arena (Giocato da: 65064)
        - Taboos: Cracks (Giocato da: 13498880, 65064)
        - Tavern Master (Giocato da: 65064)
        - Sakura Dungeon (Giocato da: 7002264)
        - Pretty Neko (Giocato da: 13498880, 65064, 7002264)
        - Sakura Hime (Giocato da: 65064)
        - Mosaique Neko Waifus 3 (Giocato da: 13498880, 7002264)
        - NEOMORPH (Giocato da: 13498880, 65064)
        - Mindustry (Giocato da: 13498880)
        - Sakura Cupid (Giocato da: 7002264)
        - Disciples: Liberation (Giocato da: 65064)
        - 60 Seconds! (Giocato da: 13498880)
        - ROMANCE OF THE THREE KINGDOMS XIV (Giocato da: 65064)
        - Control Ultimate Edition (Giocato da: 65064)
        - Sword and Fairy 7 (Giocato da: 65064)
        - TROUBLESHOOTER: Abandoned Children (Giocato da: 65064)
        - Akin Vol 2 (Giocato da: 13498880)
        - Spirit of the Island (Giocato da: 65064)
        - Crossroads Inn Anniversary Edition (Giocato da: 65064)
        - SAMURAI WARRIORS 5 (Giocato da: 65064)
        - SYNTHETIK: Legion Rising (Giocato da: 65064)
        - My Cute Fuhrer (Giocato da: 65064, 7002264)
        - Serious Sam 4 (Giocato da: 7002264)
        - West Sweety (Giocato da: 13498880, 65064)
        - Prison Simulator Prologue (Giocato da: 13498880)
        - Seek Girl Ⅲ (Giocato da: 13498880, 65064)
        - Love Tavern (Giocato da: 65064)
        - NEKOPARA Vol. 1 (Giocato da: 13498880)
        - Miss Neko 2 (Giocato da: 65064)
        - Beyond: Two Souls (Giocato da: 65064)
        - Hotline Miami (Giocato da: 7002264)
        - Chinese Brush Simulator (Giocato da: 13498880)
        - iGrow Game (Giocato da: 7002264)
        - Seen (Giocato da: 7002264)
        - 祖玛少女/Zuma Girls (Giocato da: 65064)
        - Seek Girl (Giocato da: 65064)
        - Cat Quest II (Giocato da: 13498880)
        - NEKOPARA Vol. 3 (Giocato da: 13498880, 7002264)
        - Starlight (Giocato da: 13498880, 65064)
        - LEAVES - The Return (Giocato da: 13498880)
        - OMON Simulator (Giocato da: 7002264)
        - Rebirth:Beware of Mr.Wang (Giocato da: 65064)
        - 100% Orange Juice (Giocato da: 7002264)
        - Lost (Giocato da: 13498880, 65064)
        - Orcs Must Die! (Giocato da: 13498880)
        - Yummy Girl 2 (Giocato da: 65064)
        - Hot Lava (Giocato da: 13498880)
        - Love wish (Giocato da: 13498880, 65064, 7002264)
        - King Arthur: Knight's Tale (Giocato da: 65064)
        - NEKOPARA Vol. 0 (Giocato da: 13498880)
        - Miss Neko (Giocato da: 13498880, 65064, 7002264)
        - Wasteland 3 (Giocato da: 7002264)
        - Cyber Crush 2069 (Giocato da: 65064)
        - Adorable Witch 3 (Giocato da: 65064)
        - Visitor 来访者 (Giocato da: 65064)
        - Love Fantasy (Giocato da: 13498880, 65064)
        - Tentacle Girl (Giocato da: 13498880, 65064)
        - Mosaique Neko Waifus 2 (Giocato da: 13498880, 65064, 7002264)
        - Ravenous Devils (Giocato da: 65064)
        - Seek Girl:Fog Ⅰ (Giocato da: 13498880, 65064)
        - Monster Hunter Stories 2: Wings of Ruin (Giocato da: 65064)
        - 列传：革新战争 (Giocato da: 65064)
        - Heavy Rain (Giocato da: 65064)
        - Expeditions: Rome (Giocato da: 65064)
        - Sid Meier's Civilization®: Beyond Earth™ (Giocato da: 7002264)
        - Quantum Break (Giocato da: 65064)
        - Ero Manager (Giocato da: 65064)
        - Russian Life Simulator (Giocato da: 7002264)
        - CATGIRL LOVER (Giocato da: 65064)
        - The Dungeon Of Naheulbeuk: The Amulet Of Chaos (Giocato da: 65064)
        - Mass Effect™: Andromeda Deluxe Edition (Giocato da: 65064)
        - Haydee (Giocato da: 13498880)
        - Alien Shooter (Giocato da: 13498880)
        - Cyber Agent (Giocato da: 13498880, 65064, 7002264)
        - Hacknet (Giocato da: 13498880)
        - The Last of Waifus (Giocato da: 7002264)
        - Street Fighter V (Giocato da: 65064, 7002264)
        - Melody/心跳旋律 (Giocato da: 13498880)
        - Seek Girl Ⅷ (Giocato da: 65064)
        - Chinese Parents (Giocato da: 65064)
        - Flower (Giocato da: 13498880)
        - Fairy Biography (Giocato da: 65064)
        - 200% Mixed Juice! (Giocato da: 7002264)
        - Trap Legend (Giocato da: 7002264)
        - Rabi-Ribi (Giocato da: 13498880, 7002264)
        - 99 Spirits (Giocato da: 7002264)
        - Pretty Angel (Giocato da: 13498880, 65064)
        - Sakura Beach 2 (Giocato da: 7002264)
        - Be My Girlfriends (Giocato da: 65064)
        - Immortal Life (Giocato da: 65064)
        - STAR WARS™ Empire at War - Gold Pack (Giocato da: 7002264)
        - Tales of Arise (Giocato da: 65064)
        - Eiyu*Senki – The World Conquest (Giocato da: 65064)
        - Neko Hacker Plus (Giocato da: 7002264)
        - Yakuza: Like a Dragon (Giocato da: 65064)
        - Lost Castle / 失落城堡 (Giocato da: 13498880)
        - A Wild Catgirl Appears! (Giocato da: 7002264)
        - Adorable Witch 2 (Giocato da: 7002264)
        - Good Company (Giocato da: 7002264)
        - Aimlabs (Giocato da: 13498880)
        - No Place Like Home (Giocato da: 65064)
        - Sniper Elite V2 Remastered (Giocato da: 65064)
        - Love n Dream: Virtual Happiness (Giocato da: 13498880, 65064)
        - Cute Honey 3 (Giocato da: 7002264)
        - Warhammer 40000: Dawn of War III (Giocato da: 7002264)
        - Happy Puzzle (Giocato da: 13498880, 65064)
        - Seek Girl Ⅶ (Giocato da: 13498880, 65064)
        - Swaying Girl (Giocato da: 13498880, 65064)
        - NALOGI (Giocato da: 7002264)
        - XCOM®: Chimera Squad (Giocato da: 65064)
        - Poly Bridge (Giocato da: 13498880)
        - Club Life (Giocato da: 7002264)
        - The Ditzy Demons Are in Love With Me (Giocato da: 13498880)
        - H-Rescue (Giocato da: 13498880, 65064)
        - Prominence Poker (Giocato da: 13498880)
        - Getting Over It with Bennett Foddy (Giocato da: 13498880)
        - Kinkoi: Golden Loveriche (Giocato da: 7002264)
        - Farm Together (Giocato da: 13498880)
        - Zup! F (Giocato da: 7002264)
        - Sakura Hime 2 (Giocato da: 65064)
        - HARVESTELLA (Giocato da: 65064)
        - NEKOPARA Vol. 4 (Giocato da: 13498880)
        - Wet Girl (Giocato da: 13498880, 65064, 7002264)
        - Black Legend (Giocato da: 65064)
        - OUTRIDERS (Giocato da: 65064)
        - Ken Follett's The Pillars of the Earth (Giocato da: 65064)
        - GreedFall (Giocato da: 65064)
        - CATGIRL LOVER 2 (Giocato da: 65064, 7002264)
        - Learn Japanese To Survive! Kanji Combat (Giocato da: 7002264)
        - Tower of Waifus (Giocato da: 13498880, 65064)
        - Banner of the Maid (Giocato da: 65064)
        - The Language of Love (Giocato da: 7002264)
        - 谜语女孩 (Giocato da: 65064)
        - WEED (Giocato da: 7002264)
        - Beauty Bounce (Giocato da: 7002264)
        - 恋爱关系/Romance (Giocato da: 13498880)
        - Love n Dream (Giocato da: 13498880, 65064)
        - ENDLESS™ Legend (Giocato da: 65064)
        - Cookie Clicker (Giocato da: 7002264)
        - LEAVES - The Journey (Giocato da: 13498880)
        - The Dragoness: Command of the Flame (Giocato da: 65064)
        - Happy Quest (Giocato da: 13498880, 65064)
        - NEKOPARA Extra (Giocato da: 7002264)
        - 大富翁少女/Rich Girls (Giocato da: 65064)
        - Yokai Art: Night Parade of One Hundred Demons (Giocato da: 65064)
        - Marvel's Avengers - The Definitive Edition (Giocato da: 65064)
        - NEKOPARA Vol. 2 (Giocato da: 13498880)
        - Yokai's Secret (Giocato da: 65064)
        - Winkeltje: The Little Shop (Giocato da: 65064)
        - This World Unknown (Giocato da: 7002264)
        - Seek Girl Ⅳ (Giocato da: 65064)
        - Back to Bed (Giocato da: 7002264)
        - NEKO-NIN exHeart (Giocato da: 7002264)
        - UnderMine (Giocato da: 13498880)
        - Fae Tactics (Giocato da: 65064)
        - Happy Guy (Giocato da: 65064, 7002264)
        - Farm Manager 2021 (Giocato da: 65064)
        - 3D Custom Lady Maker (Giocato da: 65064)
        - Hot And Lovely 4 (Giocato da: 13498880)

        Execution time (preprocessing): 2783 ms
        Execution time (Tf-Idf calculation): 371739 ms
        Execution time (cosine similarity calculation): 324040 ms
        Execution time (final recommendation): 268777 ms
        Execution time (total): 1072433 ms
     */


    val tFinalRecommendF = System.nanoTime()

    // Calculate and display execution times
    println(s"\n\nExecution time (preprocessing): ${(tPreProcessingF - tPreProcessingI) / 1000000} ms")
    println(s"Execution time (Tf-Idf calculation): ${(tTFIDFF - tTFIDFI) / 1000000} ms")
    println(s"Execution time (cosine similarity calculation): ${(tCosineSimilarityF - tCosineSimilarityI) / 1000000} ms")
    println(s"Execution time (final recommendation): ${(tFinalRecommendF - tFinalRecommendI) / 1000000} ms")
    println(s"Execution time (total): ${(tFinalRecommendF - tPreProcessingI) / 1000000} ms")

    spark.stop()

  }
}