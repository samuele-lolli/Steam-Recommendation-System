package com.unibo.recommendationsystem

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, collect_list, count, explode, lower, regexp_replace, split, trim, udf}

import scala.collection.parallel.CollectionsHaveToParArray
import scala.collection.parallel.immutable.ParMap
import scala.collection.parallel.mutable.ParArray

object recommendationParallel {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("recommendationsystem")
      .config("spark.master", "local[*]")
      .getOrCreate()
    //println("---------Initializing Spark-----------")

    val sc = spark.sparkContext

    val dataPathRec = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/recommendations.csv"
    val dataPathGames = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/games.csv"

    val t4 = System.nanoTime()

    val dfRec = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathRec)
    val dfGames = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathGames)

    val selectedRec = dfRec.select("app_id", "user_id", "is_recommended") //"customer_id",


    val selectedGames = dfGames.select("app_id", "title")


    val merged = selectedRec.join(selectedGames, Seq("app_id"), "inner")

    val cleanMerge = merged
      //.withColumn("title", regexp_replace(col("title"), "[^\\w\\s.,!?\\-]+", ""))
      .withColumn("title", lower(trim(regexp_replace(col("title"), "\\s+", " "))))


    val splits = cleanMerge.randomSplit(Array(0.7, 0.3), seed = 123L) // 70/30 split
    val testingDF = splits(1)



    /*
    val splits = cleanMerge.randomSplit(Array(0.7, 0.3), seed=123L) // 70/30 split
    val trainingDF = splits(0)
    val testingDF = splits(1)

     */

    val t5 = System.nanoTime()


    val t2 = System.nanoTime()

    val dataset = testingDF.withColumn("words", split(col("title"), "\\s+"))
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

    def calculateTFIDF(userWordsDataset: RDD[(String, String)]): ParArray[(String, ParMap[String, Double])] = {

      // Function to calculate term frequency (TF)
      def calculateTF(userWords: String): ParMap[String, Double] = {
        val wordsSplit = userWords.split(",")
        val totalWords = wordsSplit.size.toDouble
        val finalTF = wordsSplit.groupBy(identity).mapValues(_.length / totalWords)
        finalTF.par
      }

      // Function to calculate inverse document frequency (IDF)
      def calculateIDF(userWords: ParArray[(String, String)]): ParMap[String, Double] = {
        val userCount = userWords.length
        println("userCount:" + userCount) //userCount:22576

        val wordsCount = userWords
          .flatMap { case (user, words) =>
            words.split(",").distinct.map(word => (user, word)) // Include docId and make words distinct
          }
          .map { case (user, word) => (word, user) } // Swap for grouping by word
          .groupBy(_._1) // Group by word
          .mapValues(values => values.map(_._2).toSet.size)

        /*
        val test = wordsCount.filter(_._1.equals("battlefront"))
          test.take(1).foreach(println)
          println("battlefront:")

         */

        val idfValues = wordsCount.map { case (word, count) => (word, math.log(userCount.toDouble / count)) }.toMap[String, Double]
        idfValues
      }

      val groupedUserWords = userWordsDataset
        .groupByKey() // Group by the userId
        .map { case (userId, wordsIterable) =>
          val wordsString = wordsIterable.mkString(",")
          (userId, wordsString)
        }

      val parGroupedUserWords: ParArray[(String, String)] = groupedUserWords.collect.toParArray

      val idfValues = calculateIDF(parGroupedUserWords)

      val tfidfValues = parGroupedUserWords.map { case (user, words) =>
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

    System.out.println("TfIdfValues parallel")
    tfidfValues.take(10).foreach(println)

    /*
    (7295964,ParMap(isaac: -> 0.04342780054062241, (2007) -> 0.039119321575657846, mass -> 0.03994446377241349, post -> 0.03877663601185757, shadow -> 0.03485628113781337, any -> 0.064868037265597, years -> 0.05242703729801763, blitz -> 0.05164533024615701, hex -> 0.055603827147583045, avicii -> 0.06580971617940332, ledgehop: -> 0.08870277675652465, binding -> 0.06928508645105894, ttv3 -> 0.09685744946899459, bloodlines -> 0.05554515018862964, fight: -> 0.04466037259413426, grimm's -> 0.06094653632510593, lovecraft -> 0.059387163902023775, stick -> 0.034205143187391186, 50 -> 0.07396438889187326, majotori -> 0.0779228857932993, cut -> 0.02170449623991406, effect -> 0.04214238244263996, dead -> 0.01201850005028626, – -> 0.0331038143602397, void -> 0.043118456534355504, developer's -> 0.06103953843695432, breaker -> 0.06882653416702304, super -> 0.023025647142147502, year -> 0.020161599022636995, old -> 0.02814971954142757, edition -> 0.0036430085814957345, divinity -> 0.05572206639291835, repetendium -> 0.10162762721144358, - -> 0.005092431762970216, chrome -> 0.06076270905565792, double -> 0.04184639938809696, slay -> 0.041310736037415155, isaac -> 0.04112028042153376, minions -> 0.0734170945902863, vs -> 0.043532749549544524, button -> 0.05971626804369914, man's -> 0.03446562702124076, refunct -> 0.04912138698994983, ii: -> 0.021596186005220237, space -> 0.020292464731091846, journey -> 0.031051902953247314, lords -> 0.034563585171553154, love -> 0.034632644440021336, hollow -> 0.03975943181867001, melody -> 0.06730091271633443, masquerade -> 0.039133819058555085, castlevania: -> 0.04836634904063912, walls -> 0.06285307584668677, press -> 0.051071328314751925, laser -> 0.07645037822913452, ultimate -> 0.021144319614806593, (2008) -> 0.04246524370495294, vampire: -> 0.04438516224619655, retromaze -> 0.07424788949868573, ubermosh -> 0.05801049968571568, hades -> 0.037908798950215115, invector -> 0.06580971617940332, game -> 0.02364940417391555, tesla -> 0.060227857595931356, rebirth -> 0.03853980148942254, borderlands -> 0.023461566023083653, vampires' -> 0.07545558542880437, spire -> 0.042067685646420445, blind -> 0.037523036328768154, of -> 0.0037337232349673815, boneraiser -> 0.07577792630160571, neon -> 0.037830627859407245, the -> 0.0027974272840336294))

     */

    val t3 = System.nanoTime()

    /*val groupedTfIdf = tfidfValues
      .groupBy {case (user, _) => user } // Group by userId
      .map {case (user, mapWords) =>
        map
      }
      .mapValues(listOfMaps => listOfMaps.reduce((map1, map2) =>
        (map1 ++ map2).toArray.foldLeft(ParMap[String, Double]()) { (acc, word) =>
          acc + (word -> (map1.getOrElse(word, 0.0) + map2.getOrElse(word, 0.0)))
        }
      ))

     */


    /*groupedTfIdf.take(10).foreach(println)

(8844264,Map(homefront -> 3.8525639747220923, superflight -> 4.133005195611655, 13th: -> 3.58679622384175, raider -> 3.022166508944552, is -> 2.608198483896157, siege -> 3.018564308554879, path -> 3.1929098270117473, mafia -> 3.1183250853212114, destinies -> 4.178788368114438, seek -> 3.7295988242922786, rainbow -> 3.1124216609782023, just -> 2.8775144552961076, starve -> 3.1447657626609735, (palyaço -> 5.137419031494264, or -> 3.1934673466726244, animatronics -> 4.395361532114902, don't -> 3.0870559089730305, - -> 1.9321782769796192, evi) -> 5.137419031494264, tom -> 2.760195058769771, deceit -> 3.835922303402875, die -> 2.889894511939887, 1 -> 2.8003893925776104, six® -> 3.1218699997770183, joana's -> 5.549599479280912, (classic) -> 3.4735459338091257, cause™ -> 3.231072402168477, stories: -> 3.6829020027765824, strange -> 3.0060832704329687, bacon -> 3.684380313711222, beat -> 3.295803939367056, house -> 3.2430850999179754, tomb -> 2.7140586295829046, toxikk™ -> 4.33986111966681, case: -> 4.50985497250276, fury -> 3.6941981166422284, sniper -> 3.1003354089149044, cop -> 4.322355697777849, episode -> 2.796697800377791, life -> 2.7373640766752225, friday -> 3.563512310507262, depth -> 3.7151434513094816, clancy's -> 2.9040271215123195, game -> 2.2883465260007396, 3 -> 2.2071301910721104, munin -> 4.952087115703669, together -> 3.226634023223586, of -> 1.463991810970572, ii -> 2.422220182656557, and -> 2.389886108868645, the -> 1.3303137387817439, clown -> 4.7765041006871565, blood -> 2.938983670425039))
......
     */


    def computeCosineSimilarity(vector1: ParMap[String, Double], vector2: ParMap[String, Double]): Double = {
      def dotProduct(v1: ParMap[String, Double], v2: ParMap[String, Double]): Double = {
        v1.foldLeft(0.0) { case (acc, (key, value)) =>
          v2.get(key).map(value * _).getOrElse(0.0) + acc // Handle potential missing keys and type errors
        }
      }

      def magnitude(vector: ParMap[String, Double]): Double = {
        math.sqrt(vector.values.map(value => value * value).sum)
      }

      dotProduct(vector1, vector2) / (magnitude(vector1) * magnitude(vector2))
    }

    val targetUser = 2591067 //2821

    val t0 = System.nanoTime()

    def getSimilarUsers(userId: Int, tfidfValues: ParArray[(String, ParMap[String, Double])]): Array[(String, Double)] = {

      val userParMap = tfidfValues.find(_._1.equals(userId.toString)) match {
        case Some(entry) => Some(entry._2)
        case None => None
      }


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

      val cosineSimilarity = tfidfValues.filter(_._1 != userId.toString) // Exclude the target user
        .map { case (otherUserId, otherUserGames) =>
          (otherUserId, computeCosineSimilarity(userParMap.get, otherUserGames)) // Calculate similarity here
        }

      val sortedRDD: RDD[(String, Double)] = sc.parallelize(cosineSimilarity.seq)
        .sortBy(_._2, ascending = false) // Sort by cosineSimilarity in descending order

      sortedRDD.take(3)
    }


    val recommendations = getSimilarUsers(targetUser, tfidfValues)

    /*
    recommendations.foreach(println)

    println("Recommendations Top3")

(230202,0.4323924377136883)
(10465771,0.396194492973328)
(9381143,0.3924736687978115)
Recommendations Top3


     */

    val titlesPlayedByTargetUser = testingDF
      .filter(col("user_id") === targetUser)
      .select("title")
      .distinct()
      .collect()

    val titlesArray: Array[String] = titlesPlayedByTargetUser.map(_.getString(0))


    // 2. Extract relevant user IDs from recommendations
    val userIdsToFind = recommendations.map(_._1).toSet

    // 3. Filter datasetDF for title hasn't been played by the target user
    val filteredDF = testingDF.filter(
      col("user_id").isin(userIdsToFind.toSeq: _*) && // User ID is recommended
        !col("title").isin(titlesArray: _*) &&
        col("is_recommended") === true
    )

    val finalRecommendations = filteredDF.toDF().drop(col("is_recommended"))
      .groupBy("app_id", "title")
      .agg(collect_list("user_id").alias("users"))

    finalRecommendations.show()

    /*
+------+--------------------+-------------------+
|app_id|               title|              users|
+------+--------------------+-------------------+
|    70|           half-life|          [9381143]|
|   220|         half-life 2|          [9381143]|
|   550|       left 4 dead 2|         [10465771]|
|  3590|plants vs. zombie...|           [230202]|
|  8870|   bioshock infinite|          [9381143]|
| 12210|grand theft auto ...|          [9381143]|
| 17410|      mirror's edge™|          [9381143]|
| 32680|delta force — bla...|          [9381143]|
| 34870|sniper: ghost war...|         [10465771]|
| 35000|         mini ninjas|         [10465771]|
| 49520|       borderlands 2| [10465771, 230202]|
|203140| hitman: absolution™|          [9381143]|
|203160|         tomb raider|[10465771, 9381143]|
|219150|       hotline miami|          [9381143]|
|220240|           far cry 3|          [9381143]|
|221040|     resident evil 6|         [10465771]|
|222880|          insurgency|          [9381143]|
|223470|            postal 2|         [10465771]|
|224000|project eagle: a ...|          [9381143]|
|224480|octodad: dadliest...|         [10465771]|
+------+--------------------+-------------------+
only showing top 20 rows
 */

    val t1 = System.nanoTime()

    println("\n\nExecution time(recommendation):\t" + (t1 - t0) / 1000000 + "ms\n")
    println("\n\nExecution time(Tf-Idf calculation):\t" + (t3 - t2) / 1000000 + "ms\n")
    println("\n\nExecution time(preprocessing):\t" + (t5 - t4) / 1000000 + "ms\n")


  }
}
