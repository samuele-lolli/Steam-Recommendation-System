package com.unibo.recommendationsystem.recommender

import com.unibo.recommendationsystem.utils.timeUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

class rddRecommendation(spark: SparkSession, dataRec: Dataset[Row], dataGames: DataFrame, metadata: DataFrame) {

  /**
   * (RDD version) Generates personalized recommendations for a target user
   *
   * @param targetUser Int, The ID of the user for which we are generating recommendations
   */
  def recommend(targetUser: Int): Unit = {
    println("Preprocessing data...")
    val (appIdUserDetails, userTagsGroup, gamesData) = timeUtils.time(preprocessData(), "Preprocessing Data", "RDD")
    println("Calculate term frequency and inverse document frequency...")
    val tfidfUserTags = timeUtils.time(calculateTFIDF(userTagsGroup), "Calculating TF-IDF", "RDD")
    println("Calculate cosine similarity to get similar users...")
    val top3SimilarUsers = timeUtils.time(computeCosineSimilarity(targetUser, tfidfUserTags), "Getting Similar Users", "RDD")
    println("Calculate final recommendation...")
    timeUtils.time(generateFinalRecommendations(appIdUserDetails, top3SimilarUsers, gamesData, targetUser), "Generating Recommendations", "RDD")

    tfidfUserTags.unpersist()
    appIdUserDetails.unpersist()
    userTagsGroup.unpersist()
  }

  /**
   * Preprocesses the input data to create intermediate RDDs needed for further calculations
   *
   * @return A tuple of:
   *         - RDD[(Int, String, String)], Each app with its tags and associated user
   *         - RDD[(String, String)], Grouped tags for each user for tf-idf calculation
   *         - RDD[(Int, String)], Titles of the games
   */
  private def preprocessData(): (RDD[(Int, String, String)], RDD[(String, String)], RDD[(Int, String)]) = {

    val appIdUserId = dataRec.rdd.map(rec => (rec.getInt(0), rec.getInt(6).toString))
    /*
    (534380,22793)
    (602960,737481)
    (397540,5227423)
    (39210,5261084)
    (1196590,5935019)
    */

    //Extract tags for each appId
    val appIdTagsRDD = metadata.rdd.map(row =>
      (row.getInt(0), row.getList(2).toArray.map(_.toString).mkString(",").toLowerCase.replaceAll("\\s+", " "))
    )
    /*
    (928370,casual,sports,vr,physics,arcade,survival,cartoony,zombies,singleplayer,score attack,3d,first-person,stylized,fantasy,medieval,6dof,artificial intelligence,fps,survival horror,action)
    (1373110,immersive,casual,relaxing,vr,experimental,atmospheric,cinematic,abstract,psychedelic,exploration,free to play,simulation,colorful,immersive sim,minimalist,realistic,underwater,singleplayer)
    (42890,strategy,4x,space,sci-fi,turn-based,singleplayer,multiplayer,turn-based strategy)
    (871980,visual novel,tactical rpg,anime,jrpg,mystery,turn-based tactics,story rich,singleplayer,drama,psychological,survival,soundtrack,text-based,emotional,rpg,isometric,turn-based strategy,strategy rpg,2.5d,simulation)
    (221020,city builder,indie,sandbox,simulation,strategy,crafting,singleplayer,rpg,survival,building,resource management,isometric,pixel graphics,zombies,fantasy,management,adventure,roguelike,casual)
     */

    // Perform a join operation between appIdTagsRDD and appIdUserId RDD
    val appIdUserDetails = appIdUserId
      .join(appIdTagsRDD) // Join on appId (the key in both RDDs)
      .map { case (appId, (userId, tags)) => (appId, tags, userId) } // Reshape the result
      .persist(StorageLevel.MEMORY_AND_DISK) ////.cache()
    /*
    (1544020,horror,sci-fi,survival horror,third-person shooter,space,shooter,action-adventure,third person,pve,3d,action,story rich,linear,realistic,adventure,robots,cinematic,dark,futuristic,shoot 'em up,11608913)
    (1325200,rpg,action,souls-like,character customization,difficult,co-op,jrpg,multiplayer,fantasy,dark fantasy,historical,story rich,violent,singleplayer,hack and slash,dark,medieval,atmospheric,gore,female protagonist,11593837)
    (1549250,rpg,hack and slash,multiplayer,action rpg,action,dark fantasy,free to play,top-down,online co-op,co-op,third person,fantasy,singleplayer,loot,pve,cinematic,character customization,pvp,realistic,isometric,2326135)
    (1850570,story rich,open world,walking simulator,adventure,atmospheric,action,exploration,singleplayer,great soundtrack,third person,cinematic,post-apocalyptic,sci-fi,action-adventure,horror,relaxing,combat,multiplayer,third-person shooter,stealth,9143595)
    (1850570,story rich,open world,walking simulator,adventure,atmospheric,action,exploration,singleplayer,great soundtrack,third person,cinematic,post-apocalyptic,sci-fi,action-adventure,horror,relaxing,combat,multiplayer,third-person shooter,stealth,12014457)
    */

    //Get all tags together and remove all empty users
    val userTagsGroup = appIdUserDetails
      .flatMap { case (_, tags, userId) => tags.split(",").map(tag => (userId, tag)) }
      .reduceByKey(_ + "," + _)
      .filter(_._2.nonEmpty)
      .persist(StorageLevel.MEMORY_AND_DISK)//.cache()

    //Extracts titles of the apps to use them in ifnal recommendation
    val gamesTitles = dataGames.rdd.map(row => (row.getAs[Int]("app_id"), row.getAs[String]("title")))

    (appIdUserDetails, userTagsGroup, gamesTitles)
  }

  /**
   * Computes TF-IDF values for all users based on their tags
   *
   * @param userTagsGroup RDD[(String, String)], userId with his grouped tags
   * @return RDD[(String, Map[String, Double])], tf-idf score map for each userId
   */
  private def calculateTFIDF(userTagsGroup: RDD[(String, String)]): RDD[(String, Map[String, Double])] = {
    //TF
    def calculateTF(tags: String): Map[String, Double] = {
      val allTags = tags.split(",")
      allTags.groupBy(identity).mapValues(_.length.toDouble / allTags.size)
    }

    //IDF
    def calculateIDF(userTagsGroup: RDD[(String, String)]): Map[String, Double] = {
      val userCount = userTagsGroup.count()
      userTagsGroup.flatMap { case (_, tags) => tags.split(",").distinct }
        .map((_, 1))
        .reduceByKey(_ + _)
        .map { case (tag, count) => (tag, math.log(userCount.toDouble / count)) }
        .collect()
        .toMap
    }

    val idfValuesTag: Map[String, Double] = calculateIDF(userTagsGroup)
    /*
    (turn-based combat,2.06678519517185)
    (snow,4.500203205121008)
    (auto battler,3.9897825216318608)
    (hidden object,2.6987072144562316)
    (co-op campaign,2.6753709876189307)
     */

    //TF-IDF
    val tfidfUserTags =  userTagsGroup.map { case (user, tags) =>
      val tfValues = calculateTF(tags)
      (user, tfValues.map { case (tag, tf) => (tag, tf * idfValuesTag.getOrElse(tag, 0.0)) })
    }
    /*
    (6281566,Map(beautiful -> 0.013373348669853081, indie -> 0.0012615738716926044, superhero -> 0.018177108680027816, 2d -> 0.003646568915610358, replay value -> 0.009133181570513639, multiplayer -> 0.005210407883849194, metroidvania -> 0.013396885363901037, pvp -> 0.007576678490092771, assassin -> 0.0330417780031789, female protagonist -> 0.005446209469144816, cute -> 0.007448946878495773, action-adventure -> 0.008451774080469472, survival -> 0.008968385032315026, story rich -> 0.014832463415339302, competitive -> 0.021594502132284005, war -> 0.009668627815645125, character customization -> 0.008670864369718947, nudity -> 0.015811369590297646, stealth -> 0.030375450389445162, classic -> 0.01311370905008124, post-apocalyptic -> 0.0176879532711901, 2.5d -> 0.014657809343347736, fps -> 0.01658047037807564, survival horror -> 0.01764702102104639, dark -> 0.007541202820524046, singleplayer -> 0.0014851259093566112, puzzle -> 0.004703601467462281, modern -> 0.023096176466037746, beat 'em up -> 0.012750664032914808, heist -> 0.020069030607444395, horror -> 0.008776225110138636, steampunk -> 0.031395026441253436, shooter -> 0.011591302521908873, co-op -> 0.0029096418405069628, open world -> 0.013059586673429326, zombies -> 0.00880659192693677, difficult -> 0.011540876259976303, rpg -> 0.0028220096044712824, linear -> 0.014751976417869631, immersive sim -> 0.029153199233627903, platformer -> 0.006955310396088743, supernatural -> 0.0175381716255666, first-person -> 0.014587046908635133, atmospheric -> 0.008201243921347216, mature -> 0.008719328098575903, great soundtrack -> 0.007588237488687272, parkour -> 0.011309605228825511, based on a novel -> 0.03625905142000129, remake -> 0.026512601945725866, controller -> 0.014221852474228028, moddable -> 0.008219213192259203, multiple endings -> 0.009962230053475829, magic -> 0.02478633320442582, 2d fighter -> 0.0188563271913071, fighting -> 0.012255656663586906, dystopian -> 0.02411208106208043, violent -> 0.005655683554772282, local multiplayer -> 0.009691876553857537, arcade -> 0.008298574835029536, souls-like -> 0.015832042527138074, adventure -> 0.003348706326009017, comic book -> 0.015794567699889264, emotional -> 0.012671657744645182, military -> 0.010994111013511975, action -> 0.00332193316879919, sci-fi -> 0.004305474972966264, exploration -> 0.009943113073302146, gore -> 0.008813577807559973, online co-op -> 0.005406102970266266, fantasy -> 0.010337512813160595))
    (6030106,Map(turn-based combat -> 0.051669629879296256, indie -> 0.004415508550924116, third person -> 0.012818619704368584, 2d -> 0.012762991204636254, female protagonist -> 0.01906173314200686, mystery -> 0.06499326103423378, point & click -> 0.06840148202521243, survival -> 0.015694673806551297, story rich -> 0.007416231707669651, choices matter -> 0.028162650433610166, time travel -> 0.06740233538635025, unforgiving -> 0.0975688567214113, card game -> 0.06004904403643692, survival horror -> 0.030882286786831182, singleplayer -> 7.425629546783056E-4, casual -> 0.011385922607750868, psychological horror -> 0.023885200682403736, early access -> 0.030531623875496884, horror -> 0.015358393942742616, roguelite -> 0.04198728180013984, rpg -> 0.00987703361564949, atmospheric -> 0.00956811790823842, memes -> 0.031575523416099725, great soundtrack -> 0.006639707802601364, time manipulation -> 0.06441044091744894, lgbtq+ -> 0.055257057778127686, moddable -> 0.028767246172907213, roguelike -> 0.0395516950582583, multiple endings -> 0.0348678051871654, anime -> 0.026436518589980352, pixel graphics -> 0.022896497380035432, episodic -> 0.055921847840378264, adventure -> 0.004688188856412624, action -> 0.001660966584399595, walking simulator -> 0.0363952325525533, lovecraftian -> 0.057932201207962344))
    (2499354,Map(aliens -> 0.010941757439303201, beautiful -> 0.012317557985390996, funny -> 0.006524902867579132, rhythm -> 0.01714355101746425, indie -> 0.004647903737814858, third person -> 0.0033733209748338377, life sim -> 0.019450724468197202, cult classic -> 0.015303362693725791, 2d -> 0.010076045687870726, procedural generation -> 0.012143062527492142, mystery dungeon -> 0.0278945349315608, multiplayer -> 0.007198589839528491, female protagonist -> 0.010032491127372029, survival -> 0.012390531952540496, physics -> 0.016351023475895864, automobile sim -> 0.032734506331640635, story rich -> 0.0019516399230709604, competitive -> 0.009944836508288686, creature collector -> 0.02656516198848472, music -> 0.01313593563331314, open world survival craft -> 0.017265870497597945, stealth -> 0.006994347129148557, classic -> 0.012078416230337985, destruction -> 0.014303362850440987, post-apocalyptic -> 0.008145767953837545, 2.5d -> 0.013500613868872914, puzzle platformer -> 0.012414295892169339, building -> 0.009685872662103882, fps -> 0.003817871468635838, survival horror -> 0.008126917575481889, singleplayer -> 0.0011724678231762717, hack and slash -> 0.007831460567090885, puzzle -> 0.004332264509504732, local co-op -> 0.008452820918794221, casual -> 0.008988886269277, psychological horror -> 0.006285579126948351, early access -> 0.008034637861972864, grid-based movement -> 0.025626132701198852, horror -> 0.012125047849533643, shooter -> 0.0026690499228079643, co-op -> 0.0053598665483023, 2d platformer -> 0.01481371929866604, open world -> 0.004811426669158172, resource management -> 0.012312269635436718, difficult -> 0.007086502966652115, roguelite -> 0.02209856936849465, driving -> 0.030022711702192926, space -> 0.010022644216869797, rpg -> 0.010396877490157356, hunting -> 0.020246505962622063, simulation -> 0.007200121426525578, platformer -> 0.0064062069437659476, first-person -> 0.005374175176865575, atmospheric -> 0.007553777295977698, mature -> 0.008030960090793595, strategy -> 0.005956487660726065, great soundtrack -> 0.006989166108001435, sandbox -> 0.0047096571984768, racing -> 0.02366051493852588, touch-friendly -> 0.016526716026463526, comedy -> 0.004684636895354516, vehicular combat -> 0.019012867787706243, roguelike -> 0.020816681609609627, dungeon crawler -> 0.013581321578210404, magic -> 0.011414758712564524, 2d fighter -> 0.017367669781467067, fighting -> 0.011288104821724782, crafting -> 0.009376362505213318, top-down -> 0.010127189062644487, violent -> 0.005209182221500786, local multiplayer -> 0.017853456809737565, pixel graphics -> 0.006025394047377744, arcade -> 0.015286848380317568, adventure -> 0.0030843347739556736, turn-based -> 0.009915170915822513, perma death -> 0.014798975373308718, action -> 0.0026225788174730445, sci-fi -> 0.00793113810809575, music-based procedural generation -> 0.025765776744850175, action roguelike -> 0.013189163948151405, exploration -> 0.0045790652311259875, gore -> 0.008117769033278923, online co-op -> 0.019917221469402033, combat racing -> 0.02413388679206202, blood -> 0.027969809612324682, fantasy -> 0.01428209007081398))
    (7975745,Map(hidden object -> 0.013699021393178839, funny -> 0.005034442821685421, dark fantasy -> 0.016675958473085225, rhythm -> 0.013227511444947033, indie -> 0.003586199838314002, third person -> 0.007808296266620457, old school -> 0.009361923833051674, cyberpunk -> 0.008925587183232372, split screen -> 0.01243560884433543, side scroller -> 0.007619811121639056, cult classic -> 0.011807670707849341, 2d -> 0.00518294059071523, replay value -> 0.012981171775349335, multiplayer -> 0.0027771209533206365, assassin -> 0.02348146660124389, female protagonist -> 0.0077408053368555755, mystery -> 0.006598300612612565, physics -> 0.006308009056690789, story rich -> 0.009035003095638153, short -> 0.01340358781716095, choices matter -> 0.011436609313141183, tactical -> 0.005471779012129636, music -> 0.0101353411993076, demons -> 0.012394563128350777, precision platformer -> 0.015874312408575553, nudity -> 0.011236506307825737, crpg -> 0.01453107883896012, fast-paced -> 0.006941250772212593, stealth -> 0.016189960867470772, classic -> 0.013979080713284572, 2.5d -> 0.010416717299841031, fps -> 0.005891537697793374, survival horror -> 0.006270515083620544, dark -> 0.016077691292487817, singleplayer -> 0.0015077420399559503, hack and slash -> 0.006042548254811241, puzzle -> 0.01002798282403126, local co-op -> 0.013043947001591082, casual -> 0.006935587375279716, psychological horror -> 0.014549360821768774, isometric -> 0.00824475596423145, minimalist -> 0.01016850861272465, horror -> 0.009355366868675705, steampunk -> 0.011155593151714419, shooter -> 0.004118736936718889, co-op -> 0.004135531549959135, 2d platformer -> 0.011429874788818467, open world -> 0.009280924539492922, zombies -> 0.006258491724726639, difficult -> 0.0027338793170840646, arena shooter -> 0.01287863348736765, lore-rich -> 0.013863922180714312, space -> 0.007733207720630503, rpg -> 0.008021956236060498, immersive sim -> 0.010359004803827173, platformer -> 0.009885720359923085, supernatural -> 0.01246367526690012, first-person -> 0.006219857565103306, atmospheric -> 0.006799677701286187, mature -> 0.006196476821322977, strategy -> 0.0022979343259653855, great soundtrack -> 0.006740820104163821, sandbox -> 0.0036338471785201704, 3d vision -> 0.010401046531745513, thriller -> 0.011690627433272312, parkour -> 0.008037282903733865, free to play -> 0.0045371951892333754, based on a novel -> 0.025767853801016146, comedy -> 0.0072290843461308265, controller -> 0.015160350353238001, kickstarter -> 0.014229513960795597, surreal -> 0.008452114975119614, moddable -> 0.005841065212773037, multiple endings -> 0.021239272195227657, magic -> 0.026421979558017374, dystopian -> 0.017135489079651067, gothic -> 0.014707958714614071, turn-based strategy -> 0.009419383131203888, family friendly -> 0.011654035673614347, action rpg -> 0.012856773943198592, local multiplayer -> 0.013775256015635076, medieval -> 0.008056964374894521, adventure -> 0.003807666076274212, lego -> 0.03486582008262376, turn-based -> 0.007650284158401127, score attack -> 0.012895254082903377, action -> 0.0023607646884867338, walking simulator -> 0.014779789869057178, sci-fi -> 0.006119456814368294, crowdfunded -> 0.01748567891571525, exploration -> 0.0035330858636098993, gore -> 0.009395184464911648, fantasy -> 0.018366139515767554, lovecraftian -> 0.02352576698800501))
    (5955653,Map(funny -> 0.0247946308968007, indie -> 0.008831017101848231, third person -> 0.02563723940873717, physics -> 0.062133889208404285, short -> 0.06601266999951769, singleplayer -> 0.0014851259093566112, puzzle -> 0.032925210272235966, local co-op -> 0.06424143898283609, co-op -> 0.020367492883548742, difficult -> 0.02692871127327804, simulation -> 0.027360461420797202, great soundtrack -> 0.013279415605202727, comedy -> 0.03560324040469433, family friendly -> 0.05739612569255066, local multiplayer -> 0.06784313587700276, addictive -> 0.09984967121740512, intentionally awkward controls -> 0.21142954286906193, adventure -> 0.004688188856412624, action -> 0.00332193316879919, walking simulator -> 0.0727904651051066))
     */

    tfidfUserTags
  }


  /**
   * Computes cosine similarity between the target user and all other users
   *
   * @param targetUser Int, the ID of the target user
   * @param tfidfUserTags RDD[(String, Map[String, Double])], tf-idf score map for each userId
   * @return Array[(String, Double)], the three userId with high cosine similarity score
   */
  private def computeCosineSimilarity(targetUser: Int, tfidfUserTags: RDD[(String, Map[String, Double])]): Array[(String, Double)] = {

    val tfIdfTargetUser = tfidfUserTags.filter(_._1 == targetUser.toString).map(_._2).collect().headOption
    /*
    Map(turn-based combat -> 0.025359327548120862, rts -> 0.01092455729243764, asynchronous multiplayer -> 0.024865685359122883, indie -> 0.0043342415223795, third person -> 0.0031456735470843152, turn-based tactics -> 0.04074303642981472, replay value -> 0.015688900857324044, multiplayer -> 0.007831594672043267, female protagonist -> 0.004677725924418861, survival -> 0.007702907389718428, loot -> 0.014801822721499642, level editor -> 0.024809249615606908, war -> 0.04982605745485832, tactical -> 0.046291921826544596, third-person shooter -> 0.006654888008974728, stealth -> 0.006522335973193747, cartoon -> 0.014174409282450206, classic -> 0.005631654193286422, destruction -> 0.013338105234767056, post-apocalyptic -> 0.007596053552044829, massively multiplayer -> 0.018043623587856387, hex grid -> 0.06434338490299042, fps -> 0.014240894803255152, singleplayer -> 0.001457792303662931, hack and slash -> 0.007302957093238126, team-based -> 0.01928857576558015, historical -> 0.05180890540177583, early access -> 0.007492423036931751, isometric -> 0.00996452101198525, steampunk -> 0.013482526692562826, shooter -> 0.009955719957467744, co-op -> 0.012495394407085117, realistic -> 0.03232562611338154, open world -> 0.00448672916387756, cold war -> 0.018951439133998067, looter shooter -> 0.017285105452693333, epic -> 0.015319138101721568, mmorpg -> 0.017191568217483397, rpg -> 0.004847623860441467, simulation -> 0.016785559153863313, tanks -> 0.05811720360015172, first-person -> 0.0075172511676401925, atmospheric -> 0.003522006591989602, wargame -> 0.038699474050254694, strategy -> 0.019440806352799184, great soundtrack -> 0.0016293761478776351, sandbox -> 0.004391827571585728, free to play -> 0.005483604001711504, world war ii -> 0.07786168369510195, moddable -> 0.02117834074079059, dungeon crawler -> 0.012664790674159395, magic -> 0.010644437572452807, real time tactics -> 0.015587394925480412, top-down -> 0.009443759125901608, turn-based strategy -> 0.03415248730393557, violent -> 0.004857642316982328, action rpg -> 0.007769277505552524, adventure -> 0.001150475792984693, military -> 0.04721397367765879, turn-based -> 0.02773814685653415, action -> 0.0024455949708951093, grand strategy -> 0.01719118934026696, exploration -> 0.004270048559086811, gore -> 0.007569944129192616, online co-op -> 0.009286557249537145, fantasy -> 0.004439422680498416)
    */

    //Computes the dot product of two vectors: multiplies the target user’s score for each tag by the other user’s score for the same
    def numerator = (targetScores: Map[String, Double], otherScores: Map[String, Double]) =>
      targetScores.foldLeft(0.0) { case (acc, (tag, score)) => acc + score * otherScores.getOrElse(tag, 0.0) }

    //Computes the magnitude of a vector: computes the square root of the sum of the squares of all tf-idf values for the vector
    def denominator = (scoresMap: Map[String, Double]) => math.sqrt(scoresMap.values.map(x => x * x).sum)

    def cosineSimilarity = (target: Map[String, Double], other: Map[String, Double]) =>
      numerator(target, other) / (denominator(target) * denominator(other))

    //Finds the top 3 similar users
    val top3SimilarUsers = tfIdfTargetUser.map { targetUserScores =>
      tfidfUserTags
        .filter(_._1 != targetUser.toString)
        .map { case (userId, otherUserScores) => (userId, cosineSimilarity(targetUserScores, otherUserScores)) }
        .collect()
        .sortBy(-_._2)
        .take(3)
    }.getOrElse(Array.empty)

    /*
    (7811569,0.8584697341597517)
    (12592090,0.8123034665096692)
    (9050007,0.7995310303831706)
     */

    top3SimilarUsers
  }

  /**
   * Generates game recommendations for the target user based on similar users' preferences
   *
   * @param appIdUserDetails RDD[(Int, String, String)],  where each entry is (appId, tags, userId)
   * @param topUsersXSimilarity Array[(String, Double)], top similar users with similarity scores
   * @param gamesData RDD[(Int, String)], where each entry is (appId, game title)
   * @param targetUser Int, the ID of the target user
   */
  private def generateFinalRecommendations(appIdUserDetails: RDD[(Int, String, String)], topUsersXSimilarity: Array[(String, Double)], gamesData: RDD[(Int, String)], targetUser: Int): Unit = {

    //Gets app IDs played by the target user
    val targetUserAppIds = appIdUserDetails
      .filter(_._3 == targetUser.toString)
      .map(_._1)
      .collect()
      .toSet
    /*
    1200
    312450
    268400
    447820
    64000
    */

    val similarUsers = topUsersXSimilarity.map(_._1).toSet
    /*
    7811569
    12592090
    9050007
    */

    //Takes all games from top 3 users and removes the ones already played -> targetUserAppIds
    val finalRecommendations = appIdUserDetails
      .filter { case (appId, _, user) => similarUsers.contains(user) && !targetUserAppIds.contains(appId) }
      .map { case (appId, _, user) => (appId, user) }
      .distinct()
    /*
    (357310,9050007)
    (1550710,12592090)
    (6060,9050007)
    (1341170,12592090)
    (32470,9050007)
    ....
    */

    //Joins recommendations with game titles and prints them
    val recommendationsWithTitle = finalRecommendations
      .join(gamesData)
      .map { case (appId, (userId, title)) => (appId, title, userId) }
      .distinct()

    recommendationsWithTitle.collect().foreach { case (appId, title, userId) =>
      println(s"Game ID: $appId, userId: $userId, title: $title")
    }
    /*
    userId: 9050007, title: STAR WARS™ Empire at War - Gold Pack
    userId: 9050007, title: Oriental Empires
    userId: 9050007, title: Victory At Sea Pacific
    userId: 7811569, title: Counter-Strike: Source
    userId: 9050007, title: Tannenberg
    userId: 12592090, title: Strategic Mind: Spectre of Communism
    userId: 9050007, title: Red Orchestra 2: Heroes of Stalingrad with Rising Storm
    userId: 7811569, title: Commandos 2: Men of Courage
    userId: 12592090, title: Warbox
    userId: 9050007, title: Unity of Command II
    userId: 7811569, title: Blitzkrieg Anthology
    userId: 12592090, title: Men of War: Assault Squad 2
    userId: 12592090, title: NavalArt
    userId: 9050007, title: STAR WARS™ Republic Commando™
    userId: 9050007, title: Company of Heroes
    userId: 9050007, title: Verdun
    userId: 7811569, title: Commandos: Behind Enemy Lines
    userId: 9050007, title: Star Wars: Battlefront 2 (Classic 2005)
    userId: 7811569, title: Counter-Strike: Condition Zero
    userId: 9050007, title: Panzer Corps 2
    */
  }
}
