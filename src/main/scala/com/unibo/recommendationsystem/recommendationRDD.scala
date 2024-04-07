package com.unibo.recommendationsystem

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession.builder
import org.apache.spark.sql.functions.{col, collect_list, count, explode, lower, regexp_replace, split, trim, udf}

object recommendationRDD {
  def main(args: Array[String]): Unit = {
    val spark = builder
      .appName("recommendationsystem")
      .config("spark.master", "local[*]")
      .getOrCreate()
    //println("---------Initializing Spark-----------")

    val dataPathRec: String = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/recommendations.csv"
    val dataPathGames: String = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/games.csv"

    val t4 = System.nanoTime()

    val dfRec = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathRec)
    val dfGames = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathGames)

    val selectedRec = dfRec.select("app_id", "user_id", "is_recommended") //"customer_id",


    val selectedGames = dfGames.select("app_id", "title")



    val merged = selectedRec.join(selectedGames, Seq("app_id"), "inner")

    val cleanMerge = merged
      //.withColumn("title", regexp_replace(col("title"), "[^\\w\\s.,!?\\-]+", ""))
      .withColumn("title", lower(trim(regexp_replace(col("title"), "\\s+", " "))))

    /*
    val splits = cleanMerge.randomSplit(Array(0.7, 0.3), seed=123L) // 70/30 split
    val trainingDF = splits(0)
    val testingDF = splits(1)

     */

    val t5 = System.nanoTime()


    val t2 = System.nanoTime()

    val dataset = cleanMerge.withColumn("words", split(col("title"), "\\s+"))
    // dataset.printSchema()
    // dataset.show()

    def flattenWords = udf((s: Seq[Seq[String]]) => s.flatten)


    val aggregateData = dataset.groupBy("user_id").agg(flattenWords(collect_list("words").as("words")))

    val filteredData = aggregateData.filter(count("words") >= 20)

    //filteredData.show()
    //filteredData.printSchema()


    val explodedDF = filteredData.withColumn("word", explode(col("UDF(collect_list(words) AS words)")))
      .select("user_id", "word")

    //println("Exploded show")
    //explodedDF.show()
    /*
    +-------+----------+
|user_id|      word|
+-------+----------+
|   2821|       tom|
|   2821|  clancy's|
|   2821|     ghost|
|   2821|    recon®|
|   2821| wildlands|
|   2821|    prison|
|   2821| architect|
|   2821|       the|
|   2821| escapists|
|   2821|         2|
|   2821|assassin's|
|   2821|    creed®|
|   2821|   odyssey|
|   2821|    police|
|   2821|simulator:|
|   2821|    patrol|
|   2821|  officers|
|   2821|     ready|
|   2821|        or|
|   2821|       not|
+-------+----------+
only showing top 20 rows
     */
    val explodedRDD = explodedDF.rdd

    val userWordDataset = explodedRDD.map { row =>
      val user = row.getInt(0).toString // Assuming word is at index 0
      val word = row.getString(1) // Assuming document_frequency is at index 1
      (user, word)
    }

    def calculateTFIDF(userWordsDataset: RDD[(String, String)]): RDD[(String, Map[String, Double])] = {

      // Function to calculate term frequency (TF)
      def calculateTF(userWords: String): Map[String, Double] = {
        val wordsSplit = userWords.split(",")
        val totalWords = wordsSplit.size.toDouble
        val finalTF = wordsSplit.groupBy(identity).mapValues(_.length / totalWords)
        finalTF
      }

      // Function to calculate inverse document frequency (IDF)
      def calculateIDF(userWords: RDD[(String, String)]): Map[String, Double] = {
        val userCount = userWords.count()
        // println("userCount:" + userCount) //userCount:213364
        val wordsCount = userWords
          .flatMap { case (user, words) =>
            words.split(",").distinct.map(word => (user, word)) // Include docId and make words distinct
          }
          .map { case (user, word) => (word, user) } // Swap for grouping by word
          .groupByKey()
          .mapValues(_.toSet.size)

        /*
        val test = wordsCount.filter(_._1.equals("battlefront"))
          test.take(1).foreach(println)
          println("battlefront:")

         */

        val idfValues = wordsCount.map { case (word, count) => (word, math.log(userCount.toDouble / count)) }.collect().toMap[String,Double]
        idfValues
      }


      val groupedUserWords = userWordsDataset
        .groupByKey() // Group by the userId
        .map { case (userId, wordsIterable) =>
          val wordsString = wordsIterable.mkString(",")
          (userId, wordsString)
        }
      val idfValues = calculateIDF(groupedUserWords)

      val tfidfValues = groupedUserWords.map { case (user, words) =>
        val termFrequencies = calculateTF(words)
        val tfidf = termFrequencies.map { case (word, tf) => (word, tf * idfValues.getOrElse(word, 0.0)) }
        (user, tfidf)
      }
      tfidfValues
    }

    /*
        idfValues:
    (hoshizora,5.544142665684782)
    (dredd,4.699044625670525)
    (gaiden,4.355816950212089)
    (northend,5.611089455315395)
    (incident,4.347848020540813)
    (kishinjou,6.0882107100350575)
    (misfire,6.389240705699039)
    (serious,3.086909777014639)
    (brink,4.02188478467302)
    (sinister,4.425452878353483)

     */


    val tfidfValues = calculateTFIDF(userWordDataset)

    val t3 = System.nanoTime()

    val groupedTfIdf = tfidfValues
      .groupByKey()
      .mapValues(listOfMaps => listOfMaps.reduce((map1, map2) => map1 ++ map2))

    /*groupedTfIdf.take(10).foreach(println)

(8844264,Map(homefront -> 3.8525639747220923, superflight -> 4.133005195611655, 13th: -> 3.58679622384175, raider -> 3.022166508944552, is -> 2.608198483896157, siege -> 3.018564308554879, path -> 3.1929098270117473, mafia -> 3.1183250853212114, destinies -> 4.178788368114438, seek -> 3.7295988242922786, rainbow -> 3.1124216609782023, just -> 2.8775144552961076, starve -> 3.1447657626609735, (palyaço -> 5.137419031494264, or -> 3.1934673466726244, animatronics -> 4.395361532114902, don't -> 3.0870559089730305, - -> 1.9321782769796192, evi) -> 5.137419031494264, tom -> 2.760195058769771, deceit -> 3.835922303402875, die -> 2.889894511939887, 1 -> 2.8003893925776104, six® -> 3.1218699997770183, joana's -> 5.549599479280912, (classic) -> 3.4735459338091257, cause™ -> 3.231072402168477, stories: -> 3.6829020027765824, strange -> 3.0060832704329687, bacon -> 3.684380313711222, beat -> 3.295803939367056, house -> 3.2430850999179754, tomb -> 2.7140586295829046, toxikk™ -> 4.33986111966681, case: -> 4.50985497250276, fury -> 3.6941981166422284, sniper -> 3.1003354089149044, cop -> 4.322355697777849, episode -> 2.796697800377791, life -> 2.7373640766752225, friday -> 3.563512310507262, depth -> 3.7151434513094816, clancy's -> 2.9040271215123195, game -> 2.2883465260007396, 3 -> 2.2071301910721104, munin -> 4.952087115703669, together -> 3.226634023223586, of -> 1.463991810970572, ii -> 2.422220182656557, and -> 2.389886108868645, the -> 1.3303137387817439, clown -> 4.7765041006871565, blood -> 2.938983670425039))
......
     */

    def computeCosineSimilarity(vector1: Map[String, Double], vector2: Map[String, Double]): Double = {
      def dotProduct(v1: Map[String, Double], v2: Map[String, Double]): Double = {
        v1.foldLeft(0.0) { case (acc, (key, value)) =>
          v2.get(key).map(value * _).getOrElse(0.0) + acc // Handle potential missing keys and type errors
        }
      }

      def magnitude(vector: Map[String, Double]): Double = {
        math.sqrt(vector.values.map(value => value * value).sum)
      }

      dotProduct(vector1, vector2) / (magnitude(vector1) * magnitude(vector2))
    }

    val targetUser = 2591067 //2821

    val t0 = System.nanoTime()

    def getSimilarUsers(userId: Int, tfidfValues: RDD[(String, Map[String, Double])]): Array[(String, Double)] = {

      val userGames = tfidfValues.filter(_._1.equals(userId.toString)).first()._2

      /*
      userGames.foreach(println)  user: 2591067
      (forgotten,3.6257512547749187)
(lover,4.4821359799142675)
(sword,3.256759079319049)
(meier’s,3.305494240573331)
(maid,4.133005195611655)
(paint,3.8280636470461515)
(ⅳ,4.76052824948911)
(defeated,5.377548198026971)
(2:,2.3475427649006986)
(this,3.2121956869834856)
(giant,4.549599479280912)
(mahjong,4.07651820236299)
(dream,3.6888456809290884)
(sweety,4.684298053178368)
(point,3.697492401579769)
(is,2.608198483896157)
(delicious!,3.9943034551678047)
(4,2.261269505004634)
(符文女孩/rune,4.845830592063124)
(bioshock™,3.214978208559185)
(resilience,5.279827177347372)
(n,3.983686101775372)
(realms,3.936374490494906)
(neko,3.78069429056528)
(civilization®,3.006791335954483)
(mine,3.3011153382784943)
(remastered,2.6359990014595205)
(mortal,3.233862312074913)
(champions,3.4483203179473008)
(world,2.4829837068019205)
(human,3.070952833371827)
(xtreme,4.11423297266825)
(seek,3.7295988242922786)
(war,2.539541100149727)
(storm,2.9721874733847677)
(寇莎梅特：困世迷情,4.907244893093776)
(empires,3.1202958969957626)
(ruin,3.754879127464075)
(pretty,3.940806105293981)
(sakura,3.421640257119292)
(swaying,4.733911068583284)
(cut,2.7910450222686274)
(wingspan,4.642755843321849)
(before,3.4775130499978872)
(princess,3.8048719843842176)
(two,2.7609794441943625)
(trigger®,4.383679516078469)
(vi,3.2069027091584097)
(egypt:,4.67373471060561)
(ⅲ,4.6041582504202045)
(genius,3.9975061070407367)
(strange:,3.609318245559104)
(domination,3.8713846965355123)
(old,3.0562769970341015)
(octopath,4.229395475782318)
(edition,1.773272674496575)
(west,3.585025598359857)
(age,2.8176303917933776)
(rain,3.1114088571646867)
(iii:,3.276244481730148)
(kingdom,2.997539945093027)
(girls,3.5793042709447467)
(clicker,3.1041250129707594)
(become,3.644853888377541)
(naruto,3.448415872289139)
(elysium,3.3436361546386597)
(-,1.9321782769796192)
(banner,3.7477766947715505)
(hospital,3.7790542811322085)
(rice,4.495241816958319)
(democracy,4.271163461144358)
(wet,4.217441662012813)
(waifus,3.9585348722544125)
(v,2.7118620233690733)
(tenants,4.667359631262088)
(ⅶ,4.990291468373899)
(1,2.8003893925776104)
(happy,3.516846448995855)
(legends,3.3135803436966254)
(ⅱ,4.576985279770544)
(traveler™,4.229395475782318)
(heavy,3.9306801797051465)
(escape,3.2670214198364973)
(date,4.078756314273858)
(dragon,2.871650517500688)
(strange,3.0060832704329687)
(chance,4.6884302868449055)
(girl,3.4357489348560186)
(final,2.7307049117214355)
(solitaire,3.8682325961697734)
(by,3.279729958121827)
(love,3.29884074013096)
(seeds,4.549599479280912)
(ball,3.2148665647328882)
(fighterz,3.751793540900798)
(fairy,3.7026956384033927)
(hero,3.0851264918546413)
(consummate:missing,4.907244893093776)
(from,3.051940472733699)
(2,1.6346014658119563)
(episode,2.796697800377791)
(life,2.7373640766752225)
(7:,3.763775510092474)
(chrono,4.125671288657167)
(ultimate,2.7352593800912937)
(disco,3.475575825244252)
(catgirl,4.6138403755356)
(ⅵ,4.988630684392435)
(detroit:,3.6907616278523263)
(quest,3.100378287374919)
(idle,3.2831456470968083)
(shippuden:,3.548034018480177)
(dynasty,3.6502198155474086)
(injustice™,4.007085262999258)
(ninja,3.2276962545136993)
(breakout,4.596699457717096)
(iris,4.665780376860395)
(evil,2.4082890927008345)
(kombat 11,3.6068337955374044)
(elemental,5.0826495204144875)
(warriors,3.471626285422255)
(sid,2.9441597473015686)
(mosaique,4.118012370050454)
(of,1.463991810970572)
(and,2.389886108868645)
(sakuna:,4.61244392905301)
(definitive,2.4596894639116362)
(the,1.3303137387817439)
userGames:
       */

      tfidfValues.filter(_._1 != userId.toString) // Exclude the target user
        .map { case (otherUserId, otherUserGames) =>
          //println("other: " + otherUserId)
          (otherUserId, computeCosineSimilarity(userGames,otherUserGames)) // Calculate similarity here
        }
        .collect()
        .sortBy(-_._2)
        .take(3)
    }


    val recommendations = getSimilarUsers(targetUser, groupedTfIdf)
    recommendations.foreach(println)
    println("Recommendations Top3")
    /*
(6019065,0.7104614052219436)
(6222146,0.7004068669317693)
(8605254,0.6845741107525737)
   */


    val titlesPlayedByTargetUser = cleanMerge
      .filter(col("user_id") === targetUser)
      .select("title")
      .distinct() // In case the target user has duplicates
      .collect()

    val titlesArray: Array[String] = titlesPlayedByTargetUser.map(_.getString(0))



    // 2. Extract relevant user IDs from recommendations
    val userIdsToFind = recommendations.map(_._1).toSet

    // 3. Filter datasetDF
    val filteredDF = cleanMerge.filter(
      col("user_id").isin(userIdsToFind.toSeq: _*) && // User ID is recommended
        !col("title").isin(titlesArray: _*) &&
        col("is_recommended") === true
      // Title hasn't been played by the target user
    )

    val finalRecommendations = filteredDF.toDF().drop(col("is_recommended"))
      .groupBy("app_id", "title")
      .agg(collect_list("user_id").alias("users"))

    finalRecommendations.show()

    /*
    +-------+------------------------+--------------------+
| app_id|                   title|               users|
+-------+------------------------+--------------------+
| 750920|    shadow of the tom...|           [8605254]|
| 410900|                   forts|           [6019065]|
| 629760|                 mordhau|           [6019065]|
|1105670|          the last spell|           [6019065]|
| 636480|              ravenfield|           [6019065]|
| 450860|               andarilho|           [6222146]|
|1101450|               miss neko|[6019065, 6222146...|
| 760460|                    weed|           [8605254]|
|1182760|               starlight|           [8605254]|
| 932160|           late at night|           [8605254]|
|1013130|      happy anime puzzle|           [8605254]|
|1500750|             tauren maze|           [8605254]|
|1263370|         seek girl:fog ⅰ|  [8605254, 6019065]|
|1274610|    leslove.club: emi...|           [8605254]|
|1274300|             cyber agent|           [6222146]|
|1061010|          深海：即刻抉择|           [8605254]|
|1648470|                谜语女孩|           [8605254]|
|1397350|好久不见 - long time ...|           [8605254]|
|1299120|    mosaique neko wai...|           [6222146]|
| 385800|         nekopara vol. 0|           [6222146]|
+-------+------------------------+--------------------+
only showing top 20 rows



Execution time(preprocessing):	16032ms
     */

    val t1 = System.nanoTime()

    println("\n\nExecution time(recommendation):\t"+ (t1-t0)/1000000 + "ms\n")
    println("\n\nExecution time(Tf-Idf calculation):\t"+ (t3-t2)/1000000 + "ms\n")
    println("\n\nExecution time(preprocessing):\t"+ (t5-t4)/1000000 + "ms\n")




    // val wordsPerUser = explodedDF.groupBy("user_id")
    // .agg(count("*").alias("total_words"))

    //wordsPerUser.show()
    //wordsPerUser.printSchema()
    /*
    +-------+-----------+
|user_id|total_words|
+-------+-----------+
|   2821|        372|
|  24355|         57|
|  50405|         65|
|  56274|         75|
|  58021|         75|
|  76121|         52|
| 104853|        476|
| 108057|         58|
| 134075|         79|
| 135087|         73|
| 166338|        340|
| 197114|        177|
| 228597|         60|
| 235391|         55|
| 240130|         51|
| 242149|         95|
| 242624|         73|
| 277926|         81|
| 306142|         78|
| 322113|         79|
+-------+-----------+
only showing top 20 rows
     */




    /*
        val tf = explodedDF.groupBy("user_id", "word")
          .count()
          .withColumnRenamed("count", "term_count") // Rename to avoid ambiguity
          .join(wordsPerUser, "user_id")
          .withColumn("term_frequency", col("term_count") / col("total_words"))

        //tf.show()
        //tf.printSchema()

        /*
        +-------+---------+----------+-----------+-------------------+
    |user_id|     word|term_count|total_words|     term_frequency|
    +-------+---------+----------+-----------+-------------------+
    |   2821|      tom|         1|        366|0.00273224043715847|
    |   2821|  clancys|         1|        366|0.00273224043715847|
    |   2821|    ghost|         1|        366|0.00273224043715847|
    |   2821|    recon|         1|        366|0.00273224043715847|
    |   2821|wildlands|         1|        366|0.00273224043715847|
    |   2821|   prison|         2|        366|0.00546448087431694|
    |   2821|architect|         1|        366|0.00273224043715847|
    |   2821|      the|        17|        366|0.04644808743169399|
    |   2821|escapists|         1|        366|0.00273224043715847|
    |   2821|        2|         9|        366|0.02459016393442623|
    |   2821|assassins|         2|        366|0.00546448087431694|
    |   2821|    creed|         2|        366|0.00546448087431694|
    |   2821|  odyssey|         1|        366|0.00273224043715847|
    |   2821|   police|         1|        366|0.00273224043715847|
    |   2821|simulator|         6|        366|0.01639344262295082|
    |   2821|   patrol|         1|        366|0.00273224043715847|
    |   2821| officers|         1|        366|0.00273224043715847|
    |   2821|    ready|         1|        366|0.00273224043715847|
    |   2821|       or|         1|        366|0.00273224043715847|
    |   2821|      not|         1|        366|0.00273224043715847|
    +-------+---------+----------+-----------+-------------------+
    only showing top 20 rows

    root
     |-- user_id: integer (nullable = true)
     |-- word: string (nullable = true)
     |-- term_count: long (nullable = false)
     |-- total_words: long (nullable = false)
     |-- term_frequency: double (nullable = true)
         */

        val dfDF = explodedDF.groupBy("word")
          .agg(countDistinct("user_id").alias("document_frequency"))

        //filteredData.show()
        //filteredData.printSchema()

        /*
        +-------+---------------------------------+
    |user_id|UDF(collect_list(words) AS words)|
    +-------+---------------------------------+
    |   2821|             [tom, clancys, gh...|
    |  24355|             [resident, evil, ...|
    |  50405|             [no, mans, sky, g...|
    |  56274|             [baba, is, you, d...|
    |  58021|             [conan, exiles, d...|
    |  76121|             [subnautica, wall...|
    | 104853|             [children, of, mo...|
    | 108057|             [half-life, alyx,...|
    | 134075|             [smite, wallpaper...|
    | 135087|             [left, 4, dead, 2...|
    | 166338|             [green, hell, dis...|
    | 197114|             [resident, evil, ...|
    | 228597|             [fallout, 4, unde...|
    | 235391|             [shadowverse, ccg...|
    | 240130|             [torchlight, ii, ...|
    | 242149|             [arma, 3, beat, s...|
    | 242624|             [rust, mount, bla...|
    | 277926|             [arma, 3, path, o...|
    | 306142|             [assetto, corsa, ...|
    | 322113|             [doom, command, c...|
    +-------+---------------------------------+
    only showing top 20 rows

    root
     |-- user_id: integer (nullable = true)
     |-- UDF(collect_list(words) AS words): array (nullable = true)
     |    |-- element: string (containsNull = true)
         */

        //explodedDF.show()
        //explodedDF.printSchema()

        /*
        +-------+---------+
    |user_id|     word|
    +-------+---------+
    |   2821|      tom|
    |   2821|  clancys|
    |   2821|    ghost|
    |   2821|    recon|
    |   2821|wildlands|
    |   2821|   prison|
    |   2821|architect|
    |   2821|      the|
    |   2821|escapists|
    |   2821|        2|
    |   2821|assassins|
    |   2821|    creed|
    |   2821|  odyssey|
    |   2821|   police|
    |   2821|simulator|
    |   2821|   patrol|
    |   2821| officers|
    |   2821|    ready|
    |   2821|       or|
    |   2821|      not|
    +-------+---------+
    only showing top 20 rows

    root
     |-- user_id: integer (nullable = true)
     |-- word: string (nullable = true)
         */

        //dfDF.show()
        //dfDF.printSchema()

        /*
        +------------+------------------+
    |        word|document_frequency|
    +------------+------------------+
    |       poppy|               164|
    | battlefront|               797|
    |         few|               192|
    |   connected|               108|
    |         art|               855|
    |     blaster|               338|
    |transference|                16|
    |      online|              3394|
    |        hope|               548|
    |   firewatch|               687|
    |       still|               272|
    |      gloria|               100|
    |      waters|               256|
    |       trail|               274|
    |       anime|               247|
    |  roundabout|                33|
    |      harder|               313|
    |        some|                39|
    |    folletts|               128|
    |       those|                40|
    +------------+------------------+
    only showing top 20 rows

    root
     |-- word: string (nullable = true)
     |-- document_frequency: long (nullable = false)
         */

        val totalDocs = filteredData.select(count("user_id")).first()

        val rdd = dfDF.rdd

        val idfRDD = rdd.map { row =>
          val word = row.getString(0) // Assuming word is at index 0
          val docFreq = row.getLong(1) // Assuming document_frequency is at index 1
          val idf = math.log(totalDocs.toDouble / docFreq)
          (word, idf)
        }

        //idfRDD.take(20).foreach(println)
        /*
        (poppy,4.924776246268396)
    (battlefront,3.3437879953023804)
    (few,4.767147302064814)
    (connected,5.342511446968375)
    (art,3.2735412051558352)
    (blaster,4.201596778609576)
    (transference,7.2520539518528135)
    (online,1.8948782282984236)
    (hope,3.7183673871445797)
    (firewatch,3.492308381870246)
    (still,4.418840607796598)
    (gloria,5.419472488104504)
    (waters,4.479465229613033)
    (trail,4.4115145677045255)
    (anime,4.515254337464618)
    (roundabout,6.528135112626115)
    (harder,4.278439483552442)
    (some,6.361081027962949)
    (folletts,5.172612410172978)
    (those,6.335763219978659)
         */


        import spark.implicits._ // For toDF() method

        val idfDF = idfRDD.toDF("word", "idf")

        // Join the DataFrames on the 'word' column
        val tfidfDF = tf.join(idfDF, "word")
          .withColumn("tf_idf", col("term_frequency") * col("idf"))
          .select("user_id", "word", "tf_idf")

        val t3 = System.nanoTime()


        //tfidfDF.take(20).foreach(println)
        //tfidfDF.show()
        //tfidfDF.printSchema()

        /*
    +-------+---------+--------------------+
    |user_id|     word|              tf_idf|
    +-------+---------+--------------------+
    |   2821|      tom|0.005958453650109...|
    |   2821|  clancys|0.006004919606377874|
    |   2821|    ghost|0.006557441770309...|
    |   2821|    recon|0.009494513475024002|
    |   2821|wildlands|0.010114965667118336|
    |   2821|   prison|0.018989026950048005|
    |   2821|architect|0.009957705292876585|
    |   2821|      the|0.001569270716209...|
    |   2821|escapists|0.011471701001945752|
    |   2821|        2|0.003587925885804...|
    |   2821|assassins|0.009764474700555532|
    |   2821|    creed|0.010849646358575724|
    |   2821|  odyssey|0.007867340270752525|
    |   2821|   police|0.009216040438532427|
    |   2821|simulator|0.019833878451595316|
    |   2821|   patrol| 0.01368131922753462|
    |   2821| officers|0.015781823584817587|
    |   2821|    ready|0.011012790291765098|
    |   2821|       or|0.008031634918415432|
    |   2821|      not| 0.00719798154608242|
    +-------+---------+--------------------+
    only showing top 20 rows

    root
     |-- user_id: integer (nullable = true)
     |-- word: string (nullable = true)
     |-- tf_idf: double (nullable = true)
         */

        val aggregatedXUser = tfidfDF.groupBy("user_id")
          .agg(concat_ws(",", collect_list("word")).alias("words"),
            concat_ws(",", collect_list(col("tf_idf").cast("string"))).alias("tf_idf_values"))

        //aggregatedXUser.printSchema()
        /*
        root
     |-- user_id: integer (nullable = true)
     |-- words: string (nullable = false)
     |-- tf_idf_values: string (nullable = false)
         */
        //aggregatedXUser.show()
        //aggregatedXUser.take(10).foreach(println)

        /*
        +-------+--------------------+--------------------+
    |user_id|               words|       tf_idf_values|
    +-------+--------------------+--------------------+
    |   2821|tom,clancys,ghost...|0.005958453650109...|
    |  24355|resident,evil,rev...|0.086543583583396...|
    |  50405|no,mans,sky,green...|0.030973654845097...|
    |  58021|conan,exiles,dont...|0.052702152475653...|
    | 104853|children,of,morta...|0.008632980231637...|
    | 197114|resident,evil,rev...|0.009127006931582...|
    | 228597|fallout,4,underta...|0.027595291655757...|
    | 235391|shadowverse,ccg,t...|0.098354948313660...|
    | 242149|arma,3,beat,saber...|0.031905365526107...|
    | 277926|arma,3,path,of,ex...|0.037025979746346...|
    | 306142|assetto,corsa,bea...|0.057230269783039...|
    | 322113|doom,command,conq...|0.025314137250274...|
    | 346396|metro,exodus,hotl...|0.070605397441966...|
    | 385321|the,binding,of,is...|0.001165016393777...|
    | 387278|the,sims,4,call,o...|0.002111592213722...|
    | 394479|the,elder,scrolls...|0.001789958962625...|
    | 398679|call,to,arms,batm...|0.021134962201338...|
    | 442656|the,ascent,crusad...|0.003187309001845...|
    | 472497|borderlands,game,...|0.022799607149770...|
    | 491672|hero,siege,castle...|0.003486278947703...|
    +-------+--------------------+--------------------+
         */


        val preprocessedDF = aggregatedXUser
          .withColumn("word_array", split(col("words"), ","))
          .withColumn("tfidf_array", split(col("tf_idf_values"), ","))
          .withColumn("tfidf_array", col("tfidf_array").cast("array<double>")) // Cast to double array
          .withColumn("word_tfidf_map", map_from_arrays(col("word_array"), col("tfidf_array")))

        //  preprocessedDF.show()
        //  println("preprocessedDF:")
        //  preprocessedDF.printSchema()


        val userSimRDD = preprocessedDF.rdd.map(row => {
          val userId = row.getAs[Int]("user_id")
          val wordTfidfMap = row.getAs[Map[String, Double]]("word_tfidf_map")
          (userId, wordTfidfMap)
        })

        //   userSimRDD.take(20).foreach(println)
        /*
    (2821,Map(serious -> 0.007386486659434746, 2014 -> 0.009714186768570554, 98 -> 0.01664704109663462, masters -> 0.009774144293856235, plants -> 0.008708496883625695, legacy -> 0.006651081225382595, secret -> 0.007377466379082139, freakpocalypse -> 0.01964871401649284, directors -> 0.005694149855159431, vein -> 0.01092561939963738, defense -> 0.00648243696168337, grounded -> 0.012899829961840217, harvest -> 0.01247283377035092, prologue -> 0.007803450209914098, slobbish -> 0.020838107653809355, x -> 0.006459172711446282, festival -> 0.0159843043830518, suelle -> 0.01964871401649284, city -> 0.005635451817865926, fighters -> 0.010189553750417893, wars -> 0.004808099484401223, this -> 0.007651961381985696, spider-man -> 0.011088786707565308, demo -> 0.01210363741664141, battlefront -> 0.00913603277405022, auto -> 0.006328502750915922, raider -> 0.015690290463258774, source -> 0.007302849194084204, officers -> 0.015781823584817587, 19 -> 0.01442543864953373, force -> 0.007040686868974557, ghost -> 0.0065574417703092806, 100 -> 0.009948459123520737, creed -> 0.010849646358575724, 4 -> 0.0051956805834485, collection -> 0.005272278455707159, silent -> 0.010399023372679962, uncharted -> 0.012175678061015023, greens -> 0.01292715259390192, nazarick -> 0.022992362736771847, fire -> 0.008667759849838025, neko -> 0.033899710784915695, quartet -> 0.01964871401649284, minion -> 0.010631815931956138, cute -> 0.013068020926319543, half-life -> 0.009392959176047704, lunch -> 0.016593986217243627, survival -> 0.006597760855680439, horizon -> 0.007306356563548079, state -> 0.00866198954053982, doraemon -> 0.01907136676603599, clover -> 0.01711323103387714, mine -> 0.00812236899217678, atelier -> 0.03924754681362825, deed -> 0.01301079913985713, remastered -> 0.00872403988803207, scp -> 0.010169463656687678, another -> 0.009075012461343531, sea -> 0.007972777421763095, police -> 0.009216040438532427, blue -> 0.008682239165320262, generation -> 0.010085691382384986, memories -> 0.00925873256395068, world -> 0.0037141376082267604, wisps -> 0.010691735886857696, human -> 0.012526084436110422, happiness -> 0.016701146635695215, thieves -> 0.018766894130578738, seek -> 0.04346743710433133, war -> 0.0031571743392434758, zero -> 0.010191158125842729, jurassic -> 0.010860407701805021, cyanide -> 0.01964871401649284, friend -> 0.010448452185928703, rocksmith -> 0.014890523758440471, hand -> 0.008382779319511001, theft -> 0.006494143794130225, book -> 0.009436501944680358, nightmares -> 0.008300754684934887, detroit -> 0.011553510719473288, pretty -> 0.011681010077006398, cut -> 0.005038365372945756, battlefield -> 0.014556845743356927, f -> 0.024126616824631887, ready -> 0.011012790291765098, dead -> 0.005132780285743703, wildlands -> 0.010114965667118336, princess -> 0.010209792661013144, portal -> 0.005409880858812181, gensokyo -> 0.01590150288169844, left -> 0.012923495735125207, doom -> 0.005256487516450449, holy -> 0.012886270428118555, shop?! -> 0.015451351971654573, shift -> 0.008445955323393116, sam -> 0.007196295498242948, a -> 0.005190671893020551, dont -> 0.007058257607581908, mean -> 0.012819464155265645, night -> 0.006756640597653784, juice -> 0.011778442263888597, starve -> 0.008153295857586448, resident -> 0.022069402006694848, escapists -> 0.011471701001945752, 4.1 -> 0.01327829699499749, despair -> 0.010085691382384986, yakuza -> 0.0074821064464213475, beginning -> 0.01144762824325134, orange -> 0.011181413897936022, star -> 0.004625837241699022, galaxy -> 0.009205470829849583, old -> 0.006490235940633776, or -> 0.008031634918415432, octopath -> 0.014175849090549501, lady -> 0.012144570623241457, edition -> 0.006635555920235086, bfe -> 0.01016446413024758, underworld -> 0.012899829961840217, barro -> 0.012260405483808903, legend -> 0.006208490466640173, gas -> 0.011814751641262047, building -> 0.010985671057338952, grand -> 0.006307725156967236, girls -> 0.009091826301710685, 2023 -> 0.009939244138606421, mysterious -> 0.041319967642493925, hour -> 0.010056727431601815, dying -> 0.007223397919476718, to -> 0.003574958204756128, become -> 0.011152577319578945, doodland -> 0.019492543486875492, match -> 0.012591645551607036, miss -> 0.0136275685018551, naruto -> 0.010056727431601815, humankind -> 0.012677998937987722, alchemists -> 0.01615783827846799, code -> 0.009966982858453466, clancys -> 0.006004919606377874, - -> 0.0035457671126085948, ori -> 0.008031634918415432, playtime -> 0.012982631463193575, demon -> 0.010999197027721072, village -> 0.009051645571137747, tom -> 0.0059584536501096266, plastic -> 0.012207143749707808, homecoming -> 0.015704853975618962, waifus -> 0.012282004334921234, shooter -> 0.009463420787420827, marvels -> 0.017607912080701008, knights -> 0.006823461078466698, 6 -> 0.0064157623707744235, 1 -> 0.010136738318954888, zombies -> 0.00787597294914151, ultra -> 0.009709954017240419, odyssey -> 0.007867340270752525, shinobi -> 0.014807301880066952, blackwake -> 0.012325721114556319, raccoon -> 0.01329395457616958, zup! -> 0.016192941569301407, machine -> 0.008950500345904142, black -> 0.004662930045206896, wingdiver -> 0.019990689816913623, happy -> 0.02796113488289374, complete -> 0.010062975927310652, my -> 0.011150386422894362, hardline -> 0.015349517559640279, ink -> 0.011805628960208727, escape -> 0.01372691327550866, dragon -> 0.00574223146279071, titanfall -> 0.010071171027020253, dance -> 0.012954751216815864, iron -> 0.007888973324678169, commissar -> 0.019492543486875492, journey -> 0.007189561692155074, girlfog -> 0.017177522010407724, warfare -> 0.005640215969791681, girl -> 0.03628664829648621, guardians -> 0.010527164143939017, final -> 0.005214547110609916, backrooms -> 0.013489197647159048, teaser -> 0.011603798772868684, tomb -> 0.015585093073531552, ice -> 0.012616040717575073, tropico -> 0.009957705292876585, farming -> 0.011159757981891158, tentacle -> 0.013699473715836994, story -> 0.00557426213614863, will -> 0.008527036657125675, striker -> 0.013773322547497975, few -> 0.013024992628592387, havoc -> 0.011888850838582246, little -> 0.006581561754253448, sophie -> 0.018594991391323482, alchemist -> 0.024922463765393362, not -> 0.00719798154608242, simulator -> 0.019833878451595316, pc -> 0.010240434931013526, from -> 0.006414495978892118, 0 -> 0.015020645210634223, moonlighter -> 0.012665494383268017, 2 -> 0.0035879258858040144, episode -> 0.010453631670955494, peppa -> 0.01920467322551531, bendy -> 0.013201327385798866, ultimate -> 0.004910566030761094, firis -> 0.01981435505970714, stay -> 0.009622548164417955, prison -> 0.018989026950048005, evolution -> 0.008749850501289904, weapon -> 0.013439063388502773, traveler -> 0.013967179749608042, hunt -> 0.005863684365989708, paintings -> 0.01920467322551531, dx -> 0.025124952147201802, dynasty -> 0.010521466038772291, 7 -> 0.0057097571260858425, crosshair -> 0.01920467322551531, medieval -> 0.007620282877119796, patrol -> 0.01368131922753462, game -> 0.0026449082023455046, tannenberg -> 0.014048745222694602, king -> 0.00790203585364085, light -> 0.004742778316026411, goty -> 0.007469037967075447, 3 -> 0.005556864160622099, iv -> 0.004856345959965479, showdown -> 0.009291195208450004, we -> 0.008134222548457103, rebirth -> 0.008906021030907249, evil -> 0.018875659526923463, architect -> 0.009957705292876585, recon -> 0.009494513475024002, potatoes! -> 0.014132820792822345, dawn -> 0.005846146798723448, aer -> 0.012754252457433286, poppy -> 0.013455672804012012, laboratory -> 0.010965506829749819, danganronpa -> 0.020200578635865576, borderlands -> 0.00510810870568622, juggernaut -> 0.012292868128308229, revelations -> 0.00822848875533597, inside -> 0.008002047685778862, assassins -> 0.009764474700555532, mosaique -> 0.01329395457616958, hill -> 0.012677998937987722, of -> 0.0023845841698732933, and -> 0.014889672951855561, vs. -> 0.007415553321197193, decay -> 0.009002236621866447, lydie -> 0.01964871401649284, falsemen -> 0.021098517981143577, pig -> 0.013540269048831594, headsnatchers -> 0.015521413257483364, station -> 0.009638982742914571, overlord -> 0.011796536637551147, earth -> 0.007191243588445278, boruto -> 0.014976360360171014, seasons -> 0.01235897394411986, the -> 0.0015692707162090747, trigger -> 0.009693088281975171))
    (24355,Map(scare -> 0.0822162084789341, bioshock -> 0.07445611063323974, cyberpunk -> 0.043632729426718725, spookys -> 0.0822162084789341, half-life -> 0.030694848736013032, heroes -> 0.032538575491358177, jump -> 0.07123502217085426, dota -> 0.05518295471721494, battlefield -> 0.04756969233989852, dead -> 0.016773192719483885, у -> 0.13188545257995243, summer -> 0.05864505918725663, portal -> 0.03535743561295104, doom -> 0.034354900553944, resident -> 0.08654358358339623, симулятор -> 0.11103536043432634, edition -> 0.005421012202692056, far -> 0.03310520089884038, 2077 -> 0.04747910423005577, euro -> 0.058921589855817004, half -> 0.08261571379597933, neoncode -> 0.12841838089350677, подъезда -> 0.13188545257995243, - -> 0.007724706923897295, playtime -> 0.084850769920158, v -> 0.028203128968713333, everlasting -> 0.07540221961992022, iii -> 0.026561621360172656, truck -> 0.04899902640549739, twisted -> 0.08503486580692338, сидения -> 0.12739769493207911, simulator -> 0.02160476045620204, 2 -> 0.013027588037740767, mansion -> 0.07674633797331068, magic -> 0.05553413895310773, worlds -> 0.05453415349909053, hd -> 0.03816846586211898, 3 -> 0.03631807647835157, evil -> 0.07401955057343558, poppy -> 0.0879424329690785, revelations -> 0.05377905150808866, might -> 0.06764773364087305, of -> 0.0014168146204117296, cry -> 0.027981914203438464, infinite -> 0.035886531781148104))
     */

        def computeCosineSimilarity(vector1: Map[String, Double], vector2: Map[String, Double]): Double = {
          def dotProduct(v1: Map[String, Double], v2: Map[String, Double]): Double = {
            v1.foldLeft(0.0) { case (acc, (key, value)) =>
              v2.get(key).map(value * _).getOrElse(0.0) + acc // Handle potential missing keys and type errors
            }
          }

          def magnitude(vector: Map[String, Double]): Double = {
            math.sqrt(vector.values.map(value => value * value).sum)
          }

          dotProduct(vector1, vector2) / (magnitude(vector1) * magnitude(vector2))
        }


        val targetUser = 2591067 //2821

        val t0 = System.nanoTime()

        def getTopGamesForUserId(userId: Int, tfidfValues: RDD[(Int, Map[String, Double])]): Array[(Int, Double)] = {

          val userGames = tfidfValues.filter(_._1 == userId).first()._2

          tfidfValues.filter(_._1 != userId) // Exclude the target user
            .map { case (otherUserId, otherUserGames) =>
              (otherUserId, computeCosineSimilarity(userGames, otherUserGames)) // Calculate similarity here
            }.sortBy(-_._2)

        }
          .collect()
          .take(3)



        val recommendations = getTopGamesForUserId(targetUser, userSimRDD)

        //recommendations.foreach(println)
        //println("Recommendations Top3")
        /*
        (6019065,0.7146400338963895)
        (8605254,0.6975084350476757)
        (6222146,0.6917861806899793)
         */

        /*  6019065
        6222146
        8605254

     */
        val datasetDF = cleanMerge.toDF()

        val titlesPlayedByTargetUser = datasetDF
          .filter(col("user_id") === targetUser)
          .select("title")
          .distinct() // In case the target user has duplicates
          .as[String] // Convert DataFrame to Dataset[String]
          .collect()

        // 2. Extract relevant user IDs from recommendations
        val userIdsToFind = recommendations.map(_._1).toSet

        // 3. Filter datasetDF
        val filteredDF = datasetDF.filter(
          col("user_id").isin(userIdsToFind.toSeq: _*) && // User ID is recommended
            !col("title").isin(titlesPlayedByTargetUser: _*) &&
            col("is_recommended") === true
          // Title hasn't been played by the target user
        )

        // Display the results
        // filteredDF.show()
        /*
        +-------+--------+--------------+--------------------+
    | app_id| user_id|is_recommended|               title|
    +-------+--------+--------------+--------------------+
    | 407330| 9381143|          true|      sakura dungeon|
    |1326000| 9381143|          true|   sakura succubus 2|
    | 368500| 9381143|          true|assassins creed s...|
    |    550|10465771|          true|       left 4 dead 2|
    | 203160|10465771|          true|         tomb raider|
    | 365590| 9381143|          true|tom clancys the d...|
    | 403640| 9381143|          true|        dishonored 2|
    | 203140| 9381143|          true|   hitman absolution|
    | 227300|  230202|          true|euro truck simula...|
    | 244450|10465771|          true|men of war assaul...|
    | 257510|10465771|          true| the talos principle|
    | 319630| 9381143|          true|life is strange -...|
    | 508440|10465771|          true|totally accurate ...|
    | 952060| 9381143|          true|     resident evil 3|
    |1178830| 9381143|          true|bright memory inf...|
    |1217060|10465771|          true|      gunfire reborn|
    | 221040|10465771|          true|     resident evil 6|
    | 385800| 9381143|          true|     nekopara vol. 0|
    | 530890|10465771|          true|              haydee|
    | 836480|10465771|          true|     love simulation|
    +-------+--------+--------------+--------------------+
    only showing top 20 rows
         */

        //filteredDF.collect().foreach(println)
        //println("Recommendations for user: " + targetUser)


        //filteredDF.collect().foreach(println)


        val finalRecommendations = filteredDF.toDF().drop(col("is_recommended"))
          .groupBy("app_id", "title")
          .agg(collect_list("user_id").alias("users"))

        /*
            finalRecommendations.show()

            | app_id|                   title|               users|
        +-------+------------------------+--------------------+
        | 750920|    shadow of the tom...|           [8605254]|
        | 410900|                   forts|           [6019065]|
        | 629760|                 mordhau|           [6019065]|
        |1105670|          the last spell|           [6019065]|
        | 636480|              ravenfield|           [6019065]|
        | 450860|               andarilho|           [6222146]|
        |1101450|               miss neko|[6019065, 6222146...|
        | 760460|                    weed|           [8605254]|
        |1182760|               starlight|           [8605254]|
        | 932160|           late at night|           [8605254]|
        |1013130|      happy anime puzzle|           [8605254]|
        |1500750|             tauren maze|           [8605254]|
        |1263370|         seek girl:fog ⅰ|  [8605254, 6019065]|
        |1274610|    leslove.club: emi...|           [8605254]|
        |1274300|             cyber agent|           [6222146]|
        |1061010|          深海：即刻抉择|           [8605254]|
        |1648470|                谜语女孩|           [8605254]|
        |1397350|好久不见 - long time ...|           [8605254]|
        |1299120|    mosaique neko wai...|           [6222146]|
        | 385800|         nekopara vol. 0|           [6222146]|
         */


        finalRecommendations.collect().foreach(println)
        /*
    [750920,shadow of the tomb raider: definitive edition,WrappedArray(8605254)]
    [410900,forts,WrappedArray(6019065)]
    [629760,mordhau,WrappedArray(6019065)]
    [1105670,the last spell,WrappedArray(6019065)]
    [636480,ravenfield,WrappedArray(6019065)]
    [450860,andarilho,WrappedArray(6222146)]
    [1101450,miss neko,WrappedArray(6019065, 6222146, 8605254)]
    [760460,weed,WrappedArray(8605254)]
        */

        val t1 = System.nanoTime()

        println("\n\nExecution time(recommendation):\t"+ (t1-t0)/1000000 + "ms\n")
        println("\n\nExecution time(Tf-Idf calculation):\t"+ (t3-t2)/1000000 + "ms\n")
        println("\n\nExecution time(preprocessing):\t"+ (t5-t4)/1000000 + "ms\n")

        */
  }
}
