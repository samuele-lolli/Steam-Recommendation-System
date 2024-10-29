package com.unibo.recommendationsystem

import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel

object tagMLlib {
  def main(args: Array[String]): Unit = {

    // Configura una sessione Spark
    val spark = SparkSession.builder
      .appName("recommendationsystem")
      .config("spark.master", "local[*]")
      .config("spark.driver.memory", "6g")
      .getOrCreate()

    // Schema per il dataset recommendations.csv
    val recSchema = StructType(Array(
      StructField("app_id", IntegerType, nullable = true),
      StructField("helpful", IntegerType, nullable = true),
      StructField("funny", IntegerType, nullable = true),
      StructField("date", DateType, nullable = true),
      StructField("is_recommended", BooleanType, nullable = true),
      StructField("hours", DoubleType, nullable = true),
      StructField("user_id", IntegerType, nullable = true),
      StructField("review_id", IntegerType, nullable = true)
    ))

    // Schema per il dataset games.csv
    val gamesSchema = StructType(Array(
      StructField("app_id", IntegerType, nullable = true),
      StructField("title", StringType, nullable = true),
      StructField("date_release", DateType, nullable = true),
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

    // Schema per il dataset games_metadata.json
    val metadataSchema = StructType(Array(
      StructField("app_id", IntegerType, nullable = false),
      StructField("description", StringType, nullable = true),
      StructField("tags", ArrayType(StringType), nullable = true) // Array di stringhe per i tag
    ))

    // Path per i dataset
    val dataPathRec = "C:\\Users\\samue\\Desktop\\recommendationsystem\\steam-dataset\\recommendations.csv"
    val dataPathGames = "C:\\Users\\samue\\Desktop\\recommendationsystem\\steam-dataset\\games.csv"
    val metadataPath = "C:\\Users\\samue\\Desktop\\recommendationsystem\\steam-dataset\\games_metadata.json"

    // Inizio tempo preprocessing
    val tPreProcessingI = System.nanoTime()

    // Caricamento dei dataset in DataFrame, filtro "is_recommended = True"
    val dfRec = spark.read.format("csv").option("header", "true").schema(recSchema).load(dataPathRec).filter("is_recommended = true")
    val dfGames = spark.read.format("csv").option("header", "true").schema(gamesSchema).load(dataPathGames)
    val dfMetadata = spark.read.format("json").schema(metadataSchema).load(metadataPath)

    // Colonne utili da utilizzare
    val selectedRec = dfRec.select("app_id", "user_id")
    val selectedGames = dfGames.select("app_id", "title")

    // Join dei DataFrame selezionati su "app_id" e filtro per i giochi con tag disponibili
    val merged = selectedRec.join(selectedGames, Seq("app_id"))
      .join(dfMetadata.drop("description"), Seq("app_id"))
      .filter(size(col("tags")) > 0)

    // Preprocessing dei tag: rimuove spazi superflui e li converte in minuscolo, concatenando i tag come stringa
    val cleanMerge = merged.withColumn("tags", transform(col("tags"), tag => lower(trim(regexp_replace(tag, "\\s+", " ")))))
      .withColumn("tagsString", concat_ws(",", col("tags"))).persist(StorageLevel.MEMORY_AND_DISK).drop("tags")

    // Tokenizzazione dei tag in parole separate per il successivo calcolo di TF-IDF
    val tokenizedData = cleanMerge
      .withColumn("words", split(col("tagsString"), ","))

    // Raggruppamento dei tag per ciascun utente aggregando i tag per il calcolo di similarità del coseno
    val aggregateData = tokenizedData.groupBy("user_id")
      .agg(flatten(collect_list("words")).as("words")).persist(StorageLevel.MEMORY_AND_DISK)

    // val filteredData = aggregateData.filter(size(col("words")) >= 50)

    // Fine tempo preprocessing
    val tPreProcessingF = System.nanoTime()

    // Inizio tempo TF-IDF
    val tTFIDFI = System.nanoTime()

    // Calcolo di HashingTF per rappresentare i tag in forma vettoriale
    import spark.implicits._
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("hashedFeatures").setNumFeatures(20000)
    val featurizedData = hashingTF.transform(aggregateData).persist(StorageLevel.MEMORY_AND_DISK)

    // Applica IDF sui vettori calcolati da HashingTF per ottenere TF-IDF
    val idf = new IDF().setInputCol("hashedFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)

    // Fine tempo TF-IDF
    val tTFIDFF = System.nanoTime()

    // Inizia tempo similarità cosenica
    val tCosineSimilarityI = System.nanoTime()

    // Conversione dei vettori sparsi in vettori densi per calcoli di similarità
    val denseVector = udf { (v: Vector) => Vectors.dense(v.toArray) }
    val dfWithDenseFeatures = rescaledData.withColumn("dense_features", denseVector(col("features")))

    // Utente target per le raccomandazioni
    val targetUser = 2591067

    // Funzione UDF per calcolare similarità coseno tra l'utente target e gli altri utenti
    def cosineSimilarity(targetVector: Vector): UserDefinedFunction = udf { (otherVector: Vector) =>
      val dotProduct = targetVector.dot(otherVector)
      val normA = Vectors.norm(targetVector, 2)
      val normB = Vectors.norm(otherVector, 2)
      dotProduct / (normA * normB)
    }

    // Estrazione delle feature dell'utente target
    val targetUserFeatures = dfWithDenseFeatures.filter($"user_id" === targetUser)
      .select("features").first().getAs[Vector]("features")

    // Calcolo similarità coseno rispetto agli altri utenti
    val usersSimilarity = dfWithDenseFeatures
      .filter($"user_id" =!= targetUser)
      .withColumn("cosine_sim", cosineSimilarity(targetUserFeatures)(col("features")))
      .select("user_id", "cosine_sim")
      .orderBy($"cosine_sim".desc)
      .limit(3)

    usersSimilarity.take(3).foreach(println)
    println("Top 3 similar users:")

    // Fine tempo similarità cosenica
    val tCosineSimilarityF = System.nanoTime()

    //Inizio tempo raccomandazione finale
    val tFinalRecommendI = System.nanoTime()

    // Estrazione dei giochi giocati dall'utente target
    val titlesPlayedByTargetUser = cleanMerge.filter($"user_id" === targetUser)
      .select("tagsString").distinct().as[String].collect()

    // Estrazione dei user_id degli utenti più simili al target
    val userIdsToFind = usersSimilarity.select("user_id").as[Int].collect.toSet

    // Raccomandazioni basate su giochi giocati da utenti simili
    val finalRecommendations = cleanMerge.filter(col("user_id").isin(userIdsToFind.toArray: _*)
        && !col("title").isin(titlesPlayedByTargetUser: _*))
      .groupBy("app_id", "title")
      .agg(collect_list("user_id").alias("users"))

    finalRecommendations.take(100).foreach(println)

    // Fine tempo raccomandazione finale
    val tFinalRecommendF = System.nanoTime()

    // Ttempi di esecuzione di ciascuna sezione del codice
    println(s"\n\nExecution time (preprocessing): ${(tPreProcessingF - tPreProcessingI) / 1e9d} seconds")
    println(s"Execution time (TF-IDF): ${(tTFIDFF - tTFIDFI) / 1e9d} seconds")
    println(s"Execution time (cosine similarity): ${(tCosineSimilarityF - tCosineSimilarityI) / 1e9d} seconds")
    println(s"Execution time (final recommendation): ${(tFinalRecommendF - tFinalRecommendI) / 1e9d} seconds")
    println(s"\n\nExecution time(total): ${(tFinalRecommendF - tPreProcessingI) / 1e9d}ms\n")
    /*
    [310950,Street Fighter V,WrappedArray(7002264, 65064)]
    [1901340,Ero Manager,WrappedArray(65064)]
    [285190,Warhammer 40000: Dawn of War III,WrappedArray(7002264)]
    [899970,NEKOPARA Extra,WrappedArray(7002264)]
    [1504020,Mosaique Neko Waifus 4,WrappedArray(7002264, 13498880)]
    [966330,Flower,WrappedArray(13498880)]
    [1592110,Spirit of the Island,WrappedArray(65064)]
    [1205960,My Cute Fuhrer,WrappedArray(7002264, 65064)]
    [1431610,Prison Simulator Prologue,WrappedArray(13498880)]
    [678950,DRAGON BALL FighterZ,WrappedArray(65064)]
    [1135300,King's Bounty II,WrappedArray(65064, 7002264)]
    [1211360,NEOMORPH,WrappedArray(65064, 13498880)]
    [407330,Sakura Dungeon,WrappedArray(7002264)]
    [1732740,Sakura Hime,WrappedArray(65064)]
    [469820,Genital Jousting,WrappedArray(13498880)]
    [489020,iGrow Game,WrappedArray(7002264)]
    [372330,Beauty Bounce,WrappedArray(7002264)]
    [1205260,Defeated Girl,WrappedArray(65064)]
    [1460040,Love Fantasy,WrappedArray(65064, 13498880)]
    [521540,This World Unknown,WrappedArray(7002264)]
    [870780,Control Ultimate Edition,WrappedArray(65064)]
    [282070,This War of Mine,WrappedArray(65064)]
    [1101450,Miss Neko,WrappedArray(13498880, 7002264, 65064)]
    [1153430,Love wish,WrappedArray(7002264, 65064, 13498880)]
    [528230,SYNTHETIK: Legion Rising,WrappedArray(65064)]
    [1282410,Hard West 2,WrappedArray(65064)]
    [998930,Seek Girl,WrappedArray(65064)]
    [1148510,Pretty Angel,WrappedArray(65064, 13498880)]
    [33100,Alien Shooter,WrappedArray(13498880)]
    [404180,Club Life,WrappedArray(7002264)]
    [606800,Startup Company,WrappedArray(7002264)]
    [1201230,Immortal Life,WrappedArray(65064)]
    [1508680,Love n War: Warlord by Chance,WrappedArray(65064)]
    [1157750,CATGIRL LOVER 2,WrappedArray(65064, 7002264)]
    [1191210,Seek Girl Ⅲ,WrappedArray(65064, 13498880)]
    [728740,Sniper Elite V2 Remastered,WrappedArray(65064)]
    [949290,Winkeltje: The Little Shop,WrappedArray(65064)]
    [1677180,Neko Hacker Plus,WrappedArray(7002264)]
    [577670,Demolish & Build 2018,WrappedArray(13498880)]
    [1157390,King Arthur: Knight's Tale,WrappedArray(65064)]
    [680420,OUTRIDERS,WrappedArray(65064)]
    [1252560,Love Breakout,WrappedArray(13498880)]
    [1299120,Mosaique Neko Waifus 2,WrappedArray(13498880, 65064, 7002264)]
    [960990,Beyond: Two Souls,WrappedArray(65064)]
    [1069740,Seen,WrappedArray(7002264)]
    [1426110,Love n Dream: Virtual Happiness,WrappedArray(65064, 13498880)]
    [234270,Ken Follett's The Pillars of the Earth,WrappedArray(65064)]
    [1543030,Sword and Fairy 7,WrappedArray(65064)]
    [810670,NALOGI,WrappedArray(7002264)]
    [400910,Rabi-Ribi,WrappedArray(13498880, 7002264)]
    [970830,The Dungeon Of Naheulbeuk: The Amulet Of Chaos,WrappedArray(65064)]
    [691770,Eiyu*Senki – The World Conquest,WrappedArray(65064)]
    [1182760,Starlight,WrappedArray(65064, 13498880)]
    [384180,Prominence Poker,WrappedArray(13498880)]
    [601220,Zup! F,WrappedArray(7002264)]
    [1212830,Seek Girl Ⅳ,WrappedArray(65064)]
    [1083210,符文女孩/Rune Girl,WrappedArray(65064)]
    [1709500,Adorable Witch 2,WrappedArray(7002264)]
    [979690,The Ascent,WrappedArray(65064)]
    [1123830,Farm Manager 2021,WrappedArray(65064)]
    [382560,Hot Lava,WrappedArray(13498880)]
    [367450,Poly Bridge,WrappedArray(13498880)]
    [627270,Injustice™ 2,WrappedArray(65064)]
    [1060670,Taboos: Cracks,WrappedArray(65064, 13498880)]
    [654820,Akin Vol 2,WrappedArray(13498880)]
    [1146630,Yokai's Secret,WrappedArray(65064)]
    [429580,A Wild Catgirl Appears!,WrappedArray(7002264)]
    [1913490,Adorable Witch 3,WrappedArray(65064)]
    [1277940,Kinkoi: Golden Loveriche,WrappedArray(7002264)]
    [335190,200% Mixed Juice!,WrappedArray(7002264)]
    [1457550,Melody/心跳旋律,WrappedArray(13498880)]
    [356400,Thumper,WrappedArray(13498880)]
    [1454400,Cookie Clicker,WrappedArray(7002264)]
    [1812060,Yokai Art: Night Parade of One Hundred Demons,WrappedArray(65064)]
    [882100,XCOM®: Chimera Squad,WrappedArray(65064)]
    [1303740,Love n Dream,WrappedArray(65064, 13498880)]
    [365450,Hacknet,WrappedArray(13498880)]
    [1591530,SAMURAI WARRIORS 5,WrappedArray(65064)]
    [1605010,Adorable Witch,WrappedArray(7002264)]
    [289130,ENDLESS™ Legend,WrappedArray(65064)]
    [1070330,Russian Life Simulator,WrappedArray(7002264)]
    [1393410,Seek Girl V,WrappedArray(13498880, 65064)]
    [1509090,Seek Girl Ⅷ,WrappedArray(65064)]
    [102600,Orcs Must Die!,WrappedArray(13498880)]
    [1149660,Seek Girl Ⅱ,WrappedArray(65064, 13498880)]
    [1274300,Cyber Agent,WrappedArray(65064, 13498880, 7002264)]
    [673950,Farm Together,WrappedArray(13498880)]
    [802870,The Ditzy Demons Are in Love With Me,WrappedArray(13498880)]
    [985760,3D Custom Lady Maker,WrappedArray(65064)]
    [1226530,Trap Legend,WrappedArray(7002264)]
    [1238000,Mass Effect™: Andromeda Deluxe Edition,WrappedArray(65064)]
    [1543080,Cute Honey 3,WrappedArray(7002264)]
    [1548820,Happy Puzzle,WrappedArray(13498880, 65064)]
    [994730,Banner of the Maid,WrappedArray(65064)]
    [1615290,Ravenous Devils,WrappedArray(65064)]
    [1502230,Tower of Waifus,WrappedArray(13498880, 65064)]
    [914710,Cat Quest II,WrappedArray(13498880)]
    [606880,GreedFall,WrappedArray(65064)]
    [1140440,OMON Simulator,WrappedArray(7002264)]
    [1359650,The Last of Waifus,WrappedArray(7002264)]

    Execution time (preprocessing): 1909 ms
    Execution time (Tf-Idf calculation): 217026 ms
    Execution time (cosine similarity calculation): 30308 ms
    Execution time (final recommendation): 20087 ms
    Execution time (total): 269332 ms
    */

    spark.stop()
  }
}
