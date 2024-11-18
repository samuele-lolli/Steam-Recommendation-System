package com.unibo.recommendationsystem.utils

import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.nio.file.{Files, Paths, StandardCopyOption}
import scala.io.StdIn
import scala.jdk.CollectionConverters._
import scala.util.Random

object dataUtils {
  /**
   * Creates custom datasets for a recommendation system.
   * Asks the user whether to create new datasets or reuse an existing one.
   *
   * @param spark Spark session.
   * @return The ID of the selected target user or -1 if no user is selected.
   */
  def createCustomDatasets(spark: SparkSession, basePath: String): (Int, String) = {
    val firstChoice = promptUser("Do you want to use the full datasets or go to the customization menu? (1/2)")

    firstChoice match {
      case "1" =>
        (4893896, "full")

      case "2" =>
        val createDatasets = promptUser("Do you want to create a new custom dataset or use a custom dataset already created? (1/2)")

        val dfRec = spark.read.format("csv").option("header", "true").schema(schemaUtils.recSchema).load(basePath + "recommendations.csv").filter("is_recommended = true").sample(withReplacement = false, fraction = 0.35)
        val dfGames = spark.read.format("csv").option("header", "true").schema(schemaUtils.gamesSchema).load(basePath + "games.csv")
        val dfMetadata = spark.read.format("json").schema(schemaUtils.metadataSchema).load(basePath + "games_metadata.json")
        val dfUsers = spark.read.format("csv").option("header", "true").schema(schemaUtils.usersSchema).load(basePath + "users.csv")

        createDatasets match {
          case "1" =>
            val (userIds, targetUser) = filterUsersWithReviews(dfUsers)
            val outputDir = "steam-dataset/filteredDataset"

            saveFilteredDataset(dfRec, userIds, outputDir, "recommendations_f.csv", "user_id", "overwrite")
            val dfRecFiltered = readDataset(spark, outputDir + "/recommendations_f.csv", schemaUtils.recSchema)

            val appIds = filterAppIds(dfRecFiltered)
            saveFilteredDataset(dfGames, appIds, outputDir, "games_f.csv", "app_id", "append")
            saveFilteredDataset(dfMetadata, appIds, outputDir, "games_metadata_f.json", "app_id", "append")

            (targetUser, "custom_new")

          case "2" =>
            val targetUser = promptUser("Enter the target user ID:").toInt
            (targetUser, "custom_exist")
        }
    }
  }

  /**
   * Filters users who have at least a specified number of reviews.
   * Also allows selecting a specific target user.
   *
   * @param dfUsers DataFrame containing user information.
   * @return A tuple with the list of filtered user IDs and the target user ID.
   */
  private def filterUsersWithReviews(dfUsers: DataFrame): (List[Int], Int) = {
    val minReviews = promptUser("Enter the minimum number of reviews:").toInt
    val maxUsers = promptUser("Enter the number of users to select:").toInt
    val targetUser = promptUser("Enter the target user ID:")

    timeUtils.saveUserInput(s"MinReviews: $minReviews")
    timeUtils.saveUserInput(s"N.Users: $maxUsers")
    timeUtils.saveUserInput(s"TargetUser: $targetUser")

    val filteredUsers = dfUsers.filter(col("reviews") >= minReviews).select("user_id")
    val userList = if (targetUser.nonEmpty) {
      (filteredUsers.filter(col("user_id") === targetUser.toInt).collect().map(_.getInt(0)) ++
        filteredUsers.filter(col("user_id") =!= targetUser.toInt).limit(maxUsers - 1).collect().map(_.getInt(0))).toList
    } else {
      val allUsers = filteredUsers.limit(maxUsers).collect().map(_.getInt(0)).toList
      val randomIndex = Random.nextInt(allUsers.size)
      List(allUsers(randomIndex))
    }

    (userList, targetUser.toInt)
  }

  /**
   * Extracts a list of unique app (game) IDs from the filtered recommendations.
   *
   * @param dfRecFiltered Filtered DataFrame containing recommendations.
   * @return A list of unique app IDs.
   */
  private def filterAppIds(dfRecFiltered: DataFrame): List[Int] = {
    dfRecFiltered.select("app_id").distinct().collect().map(_.getInt(0)).toList
  }

  /**
   * Saves a filtered dataset in CSV or JSON format, depending on the specified file name.
   *
   * @param df DataFrame to save.
   * @param keys List of filter keys.
   * @param outputDir Output directory for the saved file.
   * @param fileName Name of the output file.
   * @param filterColumn Name of the column to apply the filter on.
   * @param mode Write mode (e.g., "overwrite" or "append").
   */
  private def saveFilteredDataset(df: DataFrame, keys: List[Int], outputDir: String, fileName: String, filterColumn: String, mode: String): Unit = {
    val dfFiltered = df.filter(col(filterColumn).isin(keys: _*))
    if (fileName.endsWith(".json")) saveAsSingleFile(dfFiltered, outputDir, "json", mode)
    else saveAsSingleFile(dfFiltered, outputDir, "csv", mode)

    deleteCRCFiles(outputDir)
    renamePartFile(outputDir, fileName)
  }

  /**
   * Saves a DataFrame as a single file in a specified directory, in CSV or JSON format.
   *
   * @param df DataFrame to save.
   * @param outputDir Output directory for the saved file.
   * @param format Output file format ("csv" or "json").
   * @param mode Write mode (e.g., "overwrite" or "append").
   */
  private def saveAsSingleFile(df: DataFrame, outputDir: String, format: String, mode: String): Unit = {
    df.coalesce(1)
      .write
      .format(format)
      .option("header", format == "csv")
      .option("delimiter", if (format == "csv") "," else null)
      .mode(mode)
      .save(outputDir)
  }

  /**
   * Renames the generated part-* output file to the specified file name.
   *
   * @param outputDir Directory containing the part-* file.
   * @param targetFileName Final name to assign to the file.
   */
  private def renamePartFile(outputDir: String, targetFileName: String): Unit = {
    val path = Paths.get(outputDir)
    val extension = if (targetFileName.endsWith(".json")) ".json" else ".csv"

    Files.list(path)
      .iterator()
      .asScala
      .find(p => p.getFileName.toString.startsWith("part-") && p.getFileName.toString.endsWith(extension))
      .foreach { partFile =>
        Files.move(partFile, path.resolve(targetFileName), StandardCopyOption.REPLACE_EXISTING)
        println(s"File renamed to $targetFileName")
      }
  }

  /**
   * Deletes .crc checksum files in the specified directory.
   *
   * @param dirPath Path to the directory where .crc files are located.
   */
  private def deleteCRCFiles(dirPath: String): Unit = {
    Files.walk(Paths.get(dirPath))
      .iterator()
      .asScala
      .filter(_.toString.endsWith(".crc"))
      .foreach(Files.delete)
  }

  /**
   * Reads a CSV dataset with the specified schema.
   *
   * @param spark Spark session.
   * @param path Path to the CSV file.
   * @param schema Schema to apply while reading the dataset.
   * @return Resulting DataFrame.
   */
  private def readDataset(spark: SparkSession, path: String, schema: org.apache.spark.sql.types.StructType): DataFrame = {
    spark.read.format("csv").option("header", "true").schema(schema).load(path)
  }

  /**
   * Displays a message to the user and reads input.
   *
   * @param message Message to display.
   * @return User input as a lowercase trimmed string.
   */
  private def promptUser(message: String): String = {
    println(message)
    StdIn.readLine().trim.toLowerCase
  }
}