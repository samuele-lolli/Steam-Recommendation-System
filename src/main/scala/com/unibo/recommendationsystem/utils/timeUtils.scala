package com.unibo.recommendationsystem.utils

import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ListBuffer

object timeUtils {

  //val logFilePath: String = "gs://dataproc-staging-us-central1-534461255477-conaqzw0/output/"
  val logFilePath = "timing_log.txt"
  private val logBuffer = new ListBuffer[String]()

  // Method to add log messages to the buffer
  private def logToBuffer(message: String): Unit = {
    logBuffer += message
  }

  def time[R](block: => R, operation: String = "unknown", className: String = "UnknownClass"): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()

    val elapsedMillis = (t1 - t0) / 1000000
    val elapsedSeconds = (t1 - t0) / 1e9

    val logMessage = f"[$className] Elapsed time for $operation: $elapsedMillis%d ms ($elapsedSeconds%.3f s)"
    logToBuffer(logMessage)
    println(logMessage)

    result
  }


  // Method to flush all buffered logs to GCS as a single file
  def flushLogsToGCS(spark: SparkSession): Unit = {
    import spark.implicits._

    // Convert logBuffer to an RDD and write to GCS as a single file
    val logRDD = spark.sparkContext.parallelize(logBuffer)
    logRDD.coalesce(1).saveAsTextFile(logFilePath)
  }
}