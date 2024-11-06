package com.unibo.recommendationsystem.utils

import org.apache.spark.sql.SparkSession

object timeUtils {

  val logFilePath: String = "gs://dataproc-staging-us-central1-534461255477-conaqzw0/output/timing_log"

  private def logToFile(spark: SparkSession, message: String, filePath: String = logFilePath): Unit = {
    // Create an RDD and save to GCS
    val logRDD = spark.sparkContext.parallelize(Seq(message))
    val timestamp = System.currentTimeMillis()
    logRDD.saveAsTextFile(s"$filePath/log_$timestamp")
  }

  def time[R](spark: SparkSession, block: => R, operation: String = "unknown", className: String = "UnknownClass", filePath: String = logFilePath): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    val elapsedMillis = (t1 - t0) / 1000000
    val elapsedNanos = t1 - t0
    val logMessage = s"[$className] Elapsed time for $operation:\t${elapsedMillis}ms (${elapsedNanos}ns)"
    logToFile(spark, logMessage, filePath)
    println(logMessage)
    result
  }

  def roundAt(p: Int, n: Double): Double = {
    val s = math.pow(10, p)
    math.round(n * s) / s
  }
}
