package com.unibo.recommendationsystem.utils

import com.google.cloud.storage.{BlobId, BlobInfo, Storage, StorageOptions}
import java.nio.charset.StandardCharsets

object timeUtils {

  val bucketName: String = "your-gcs-bucket-name" // Replace with your actual GCS bucket name
  val logFilePath: String = "timing_log.txt" // GCS path within the bucket

  // Initialize GCS storage client
  private val storage: Storage = StorageOptions.getDefaultInstance.getService

  private def logToFile(message: String, bucket: String = bucketName, gcsFilePath: String = logFilePath): Unit = {
    val blobId = BlobId.of(bucket, gcsFilePath)
    val blobInfo = BlobInfo.newBuilder(blobId).build()

    // Append new message to the log content
    val newContent = message + "\n"
    val existingContent = Option(storage.get(blobId)).map(blob => new String(blob.getContent(), StandardCharsets.UTF_8)).getOrElse("")
    val updatedContent = existingContent + newContent

    // Write updated content back to GCS
    storage.create(blobInfo, updatedContent.getBytes(StandardCharsets.UTF_8))
  }

  def time[R](block: => R, operation: String = "unknown", className: String = "UnknownClass", filePath: String = logFilePath): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    val elapsedMillis = (t1 - t0) / 1000000
    val elapsedNanos = t1 - t0
    val logMessage = s"[$className] Elapsed time for $operation:\t${elapsedMillis}ms (${elapsedNanos}ns)"
    logToFile(logMessage, bucketName, filePath)
    println(logMessage)
    result
  }

  def roundAt(p: Int, n: Double): Double = {
    val s = math.pow(10, p)
    math.round(n * s) / s
  }
}
