package com.unibo.recommendationsystem.utils

import org.apache.spark.sql.types._

object schemaUtils {
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

  val metadataSchema: StructType = StructType(Array(
    StructField("app_id", IntegerType, nullable = false),
    StructField("description", StringType, nullable = true),
    StructField("tags", ArrayType(StringType), nullable = true)
  ))

  val usersSchema: StructType = StructType(Array(
    StructField("user_id", IntegerType, nullable = false),
    StructField("products", IntegerType, nullable = true),
    StructField("reviews", IntegerType, nullable = true)
  ))
}
