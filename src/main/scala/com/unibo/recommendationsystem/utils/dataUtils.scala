package com.unibo.recommendationsystem.utils

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, concat_ws}

import java.nio.file.{Files, Paths, StandardCopyOption}
import scala.io.StdIn
import scala.jdk.CollectionConverters._
import scala.util.Random

object dataUtils {

  def createCustomDatasets(spark: SparkSession, dfRec: DataFrame, dfGames: DataFrame, dfMetadata: DataFrame, dfUsers: DataFrame): Int = {
    println("Do you want to create custom datasets? (y/n)")
    val createDatasets = StdIn.readLine().toLowerCase

    if (createDatasets == "y") {
      val (userIds, targetUser) = filterUsersWithReviews(dfUsers)
      val outputDir = "steam-dataset/filteredDataset"
      val targetFileName = "recommendations_f.csv"
      saveFilteredDataset(dfRec, userIds, outputDir, targetFileName, "user_id", "overwrite")

      val dfRecFiltered = spark.read.format("csv").option("header", "true").schema(schemaUtils.recSchema).load(outputDir + "/" + targetFileName)
      val appIds = filterAppIds(dfRecFiltered)

      saveFilteredDataset(dfGames, appIds, outputDir, "games_f.csv", "app_id", "append")

      val convertMetadata = dfMetadata.select(dfMetadata.columns.map {
        case colName if dfMetadata.schema(colName).dataType.typeName == "array" =>
          concat_ws(",", dfMetadata(colName)).alias(colName) // Join array elements with a comma or other delimiter
        case colName => dfMetadata(colName)
      }: _*)
      saveFilteredDataset(convertMetadata, appIds, outputDir, "games_metadata_f.csv", "app_id", "append")

      targetUser
    } else {
      println("Do you want to use a custom datasets already created? (y/n)")
      val reuseDataset = StdIn.readLine().toLowerCase
      if (reuseDataset == "y") {
        println("Enter the target user ID:")
        val targetUser = StdIn.readLine()

        targetUser.toInt
      } else {
        println("Skipping custom dataset creation.")
        -1
      }
      // Handle the case where the user chooses not to create datasets
       // Return a dummy value (e.g., -1) to indicate no creation
    }
  }

  /**
   * Filters users with custom reviews and selects up to "n" users.
   *
   * @param dfUsers DataFrame of users with reviews column.
   * @return List of user IDs that meet the criteria.
   */
  def filterUsersWithReviews(dfUsers: DataFrame): (List[Int], Int) = {
    println("Enter the minimum number of reviews:")
    val minReviews = StdIn.readInt()

    println("Enter the number of users to select:")
    val maxUsers = StdIn.readInt()

    println("Enter the target user ID:")
    val targetUser = StdIn.readLine()

    val filteredUsers = dfUsers.filter(col("reviews") >= minReviews).select("user_id")

    // Limit the number of users and add the target user if necessary
    val userList = if (targetUser.nonEmpty) {
      filteredUsers.filter(col("user_id") === targetUser.toInt).collect().map(_.getInt(0)).toList ++
        filteredUsers.filter(col("user_id") =!= targetUser.toInt).limit(maxUsers - 1).collect().map(_.getInt(0)).toList
    } else {
      val allUsers = filteredUsers.limit(maxUsers).collect().map(_.getInt(0)).toList
      val randomIndex = Random.nextInt(allUsers.size)
      val selectedUser = allUsers(randomIndex)
      List(selectedUser)
    }
    (userList, targetUser.toInt)
  }

  /**
   * Filters users with at least 10 reviews and selects up to 100 users.
   *
   * @param dfRecFiltered DataFrame of filtered users with reviews and app_id columns.
   * @return List of user IDs that meet the criteria.
   */
  def filterAppIds(dfRecFiltered: DataFrame): List[Int] = {
    dfRecFiltered.select("app_id").distinct().collect().map(_.getInt(0)).toList
  }

  /**
   * Filters a DataFrame by a list of user IDs and saves it as a single CSV file.
   *
   * @param dfToFilter DataFrame to filter and save.
   * @param keysId List of key IDs to filter by.
   * @param outputDir Directory to save the filtered CSV file.
   */
  def filterAndSaveDataset(dfToFilter: DataFrame, keysId: List[Int], outputDir: String, filterColumn: String, mode: String): Unit = {
    val dfFiltered = dfToFilter.filter(col(filterColumn).isin(keysId: _*))
    saveAsSingleCSV(dfFiltered, outputDir, mode)
  }


  /**
   * Saves a DataFrame as a single CSV file in the specified directory.
   *
   * @param df DataFrame to save.
   * @param outputDir Directory to save the CSV file.
   */
  def saveAsSingleCSV(df: DataFrame, outputDir: String, mode: String): Unit = {
    df.coalesce(1)
      .write
      .format("csv")
      .option("header", "true")
      .option("delimiter", ",")
      .mode(mode)
      .save(outputDir)
  }

  /**
   * Deletes all .crc files in the specified directory.
   *
   * @param dirPath Path to the directory.
   */
  def deleteCRCFiles(dirPath: String): Unit = {
    Files.walk(Paths.get(dirPath))
      .iterator()
      .asScala
      .filter(path => path.toString.endsWith(".crc"))
      .foreach(Files.delete)
  }

  /**
   * Renames the part file in the specified directory to a target filename.
   *
   * @param dirPath Path to the directory.
   * @param targetFileName Desired filename for the CSV file.
   */
  def renamePartFile(dirPath: String, targetFileName: String): Unit = {
    val path = Paths.get(dirPath)
    val partFile = Files.list(path)
      .iterator()
      .asScala
      .find(p => p.getFileName.toString.startsWith("part-") && p.getFileName.toString.endsWith(".csv"))

    partFile match {
      case Some(p) =>
        val targetPath = path.resolveSibling("filteredDataset/"+targetFileName)
        Files.move(p, targetPath, StandardCopyOption.REPLACE_EXISTING)
        println(s"File renamed to $targetFileName")
      case None =>
        println("No part file found to rename.")
    }
  }

  /**
   * Combines filtering, saving as CSV, deleting .crc files, and renaming the output file.
   *
   * @param dfToFilter DataFrame to filter.
   * @param keysId List of user IDs to filter by.
   * @param outputDir Directory to save the CSV file.
   * @param targetFileName Desired filename for the final CSV file.
   */
  def saveFilteredDataset(dfToFilter: DataFrame, keysId: List[Int], outputDir: String, targetFileName: String, filterColumn: String, mode: String): Unit = {
    // Filter and save the dataset
    filterAndSaveDataset(dfToFilter, keysId, outputDir, filterColumn, mode)
    renamePartFile(outputDir, targetFileName)

    // Delete any .crc files
    deleteCRCFiles(outputDir)

    // Rename the part file to the target file name
  }

}
