package com.unibo.recommendationsystem.utils

import org.apache.spark.sql.types._

object schemaUtils {
  /**
   * Schema for the recommendations dataset.
   * Contains information about user reviews for specific apps.
   *
   * Fields:
   * - `app_id`: Integer, the unique identifier for the app (non-nullable).
   * - `helpful`: Integer, the number of helpful votes for the review.
   * - `funny`: Integer, the number of funny votes for the review.
   * - `date`: String, the date of the review.
   * - `is_recommended`: Boolean, whether the app is recommended by the user.
   * - `hours`: Double, the number of hours the user spent on the app.
   * - `user_id`: Integer, the unique identifier for the user (non-nullable).
   * - `review_id`: Integer, the unique identifier for the review (non-nullable).
   */
  val recSchema: StructType = StructType(Array(
    StructField("app_id", IntegerType, nullable = false),
    StructField("helpful", IntegerType, nullable = true),
    StructField("funny", IntegerType, nullable = true),
    StructField("date", StringType, nullable = true),
    StructField("is_recommended", BooleanType, nullable = true),
    StructField("hours", DoubleType, nullable = true),
    StructField("user_id", IntegerType, nullable = false),
    StructField("review_id", IntegerType, nullable = false)
  ))

  /**
   * Schema for the games dataset.
   * Contains information about the games and their properties.
   *
   * Fields:
   * - `app_id`: Integer, the unique identifier for the game (non-nullable).
   * - `title`: String, the title of the game.
   * - `date_release`: String, the release date of the game.
   * - `win`: Boolean, whether the game supports Windows.
   * - `mac`: Boolean, whether the game supports macOS.
   * - `linux`: Boolean, whether the game supports Linux.
   * - `rating`: String, the rating of the game.
   * - `positive_ratio`: Integer, the percentage of positive reviews.
   * - `user_reviews`: Integer, the total number of user reviews.
   * - `price_final`: Double, the final price of the game after discounts.
   * - `price_original`: Double, the original price of the game.
   * - `discount`: Double, the discount percentage on the game.
   * - `steam_deck`: Boolean, whether the game is compatible with the Steam Deck.
   */
  val gamesSchema: StructType = StructType(Array(
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

  /**
   * Schema for the metadata dataset.
   * Contains additional metadata about the games.
   *
   * Fields:
   * - `app_id`: Integer, the unique identifier for the game (non-nullable).
   * - `description`: String, the description of the game.
   * - `tags`: Array of Strings, the tags associated with the game.
   */
  val metadataSchema: StructType = StructType(Array(
    StructField("app_id", IntegerType, nullable = false),
    StructField("description", StringType, nullable = true),
    StructField("tags", ArrayType(StringType), nullable = true)
  ))

  /**
   * Schema for the users dataset.
   * Contains information about the users.
   *
   * Fields:
   * - `user_id`: Integer, the unique identifier for the user (non-nullable).
   * - `products`: Integer, the number of products owned by the user.
   * - `reviews`: Integer, the number of reviews written by the user.
   */
  val usersSchema: StructType = StructType(Array(
    StructField("user_id", IntegerType, nullable = false),
    StructField("products", IntegerType, nullable = true),
    StructField("reviews", IntegerType, nullable = true)
  ))
}
