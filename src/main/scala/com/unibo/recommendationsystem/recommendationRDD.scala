package com.unibo.recommendationsystem

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession.builder

object recommendationRDD {
  def main(args: Array[String]): Unit = {
    val spark = builder
      .appName("sparkRecommend")
      .config("spark.master", "local[*]")
      .getOrCreate()
    //println("---------Initializing Spark-----------")

    val dataPathRec = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/recommendations.csv"
    val dataPathGames = "/Users/leonardovincenzi/IdeaProjects/recommendationsystem/steam-dataset/games.csv"

    val t4 = System.nanoTime()

    val dfRec = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathRec)
    val dfGames = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(dataPathGames)


    val rddRec = dfRec.rdd
    val rddGames = dfGames.rdd


    val selectedRecRDD = rddRec.map { row =>
      val appId = row.getInt(0)
      val recommended = row.getBoolean(4)
      val user = row.getInt(6).toString
      (appId, recommended, user)
    }

    val selectedGamesRDD = rddGames.map { row =>
      val appId = row.getInt(0)
      val title = row.getString(1)
      (appId, title)
    }

    val titleDict = selectedGamesRDD.collectAsMap()

    val mergedRDD = selectedRecRDD.map {
      case  (appId, recommended , user) =>
        val title = titleDict.getOrElse(appId, "") // Handle potential missing titles
        (appId, recommended, user, title)
    }


    def transformTitle(row: (Int, Boolean, String, String)): (Int, Boolean, String, String) = {
      val appId = row._1
      val recommended = row._2
      val user = row._3
      var title = row._4

      // Apply transformations to the title
      title = title.toLowerCase()
        .trim()
        .replaceAll("\\s+", " ")

      (appId, recommended, user, title)
    }

    val cleanMergeRDD = mergedRDD.map(transformTitle)

    val t5 = System.nanoTime()


    val t2 = System.nanoTime()


    def splitTitle(row: (Int, Boolean, String, String)): (Int, Boolean, String, Array[String]) = {
      val appId = row._1
      val recommended = row._2
      val user = row._3
      val title = row._4

      val words = title.split("\\s+")
      (appId, recommended, user, words)
    }

    val datasetRDD = cleanMergeRDD.map(splitTitle)

    def flattenWordsRDD(wordLists: Seq[Array[String]]): Array[String] = {
      wordLists.flatten.toArray
    }

    // Perform the aggregation
    val aggregateDataRDD = datasetRDD
      .map { case (_, _, user, words) => (user, words) } // Map to (user, words)
      .groupByKey() // Group by user
      .mapValues { wordsIterable =>
        flattenWordsRDD(wordsIterable.toSeq) // Flatten words for each user
      }

    val prova = aggregateDataRDD.filter{ case ( userId, _) => userId.equals("2821")}
    prova.collect().foreach { case (key, wordArray) =>
      println(s"Key: $key")
      println("Array:")
      wordArray.foreach(println)
    }
    // println("aggregateDataRDD: " + aggregateDataRDD.count()) //aggregateDataRDD: 13781059
    // println("filteredDataRDD: " + filteredDataRDD.count()) //filteredDataRDD:     1178775

    val filteredDataRDD = aggregateDataRDD.filter { case (_, words) => words.length >= 20 }

    println("filteredDataRDD: " + filteredDataRDD.keys.count())
    //filteredData: 1178775


    val explodedRDD = aggregateDataRDD.flatMap { case (userId, words) =>
      words.map(word => (userId, word))
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

        //userCount:1178775

        println("userCount:" + userCount)
        val wordsCount = userWords
          .flatMap { case (user, words) =>
            words.split(",").distinct.map(word => (user, word)) // Include docId and make words distinct
          }
          .map { case (user, word) => (word, user) } // Swap for grouping by word
          .groupByKey()
          .mapValues(_.toSet.size)


        val test = wordsCount.filter(_._1.equals("battlefront"))
        test.take(1).foreach(println)
        println("battlefront:")



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


    val tfidfValues = calculateTFIDF(explodedRDD)

    tfidfValues.take(20).foreach(println)
    println("tfidfValues")

    val t3 = System.nanoTime()


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
        .take(10)
    }


    val recommendations = getSimilarUsers(targetUser, tfidfValues)
    recommendations.foreach(println)
    println("Recommendations Top3")
    /*
(6019065,0.7104614052219436)
(6222146,0.7004068669317693)
(8605254,0.6845741107525737)


    /*
(10941911,0.7293625797795579)
(14044364,0.7263267622929318)
(4509885,0.7186991307198306)
(3278010,0.7159065615500113)
(6019065,0.7126999191199811)
(7889674,0.7113882151776377)
(1216253,0.7088144049757779)
(144350,0.7063527142603677)
(6222146,0.7033717175918999)
(10974221,0.7028838351034404)
     */

   */


    val titlesPlayedByTargetUserRDD = cleanMergeRDD
      .filter { case (_, _, user, _) => user.equals(targetUser.toString)}
      .map { case (_, _, _, title) => title }
      .distinct()

    val titlesPlayedByTargetUserSet = titlesPlayedByTargetUserRDD.collect().toSet


    val userIdsToFind = recommendations.take(3).map(_._1).toSet

    val filteredRDD = cleanMergeRDD.filter { case (_, isRecommended, user, title) =>
      userIdsToFind.contains(user) &&
        !titlesPlayedByTargetUserSet.contains(title) &&
        isRecommended
    }

    filteredRDD.take(10).foreach(println)
    println("filteredRDD")

    val finalRecommendations = filteredRDD.map { case (appId, _, user, title) =>
      ((appId, title), user)
    }



    val t1 = System.nanoTime()

    finalRecommendations.take(20).foreach(println)


    println("\n\nExecution time(recommendation):\t"+ (t1-t0)/1000000 + "ms\n")
    println("\n\nExecution time(Tf-Idf calculation):\t"+ (t3-t2)/1000000 + "ms\n")
    println("\n\nExecution time(preprocessing):\t"+ (t5-t4)/1000000 + "ms\n")

  }
}

