package com.unibo.recommendationsystem.utils

import java.io.File
import java.nio.file.{Files, Paths, StandardOpenOption}

object timeUtils {

  val logFilePath: String = "timing_log.txt"

  private def logToFile(message: String, filePath: String = logFilePath): Unit = {
    val file = new File(filePath)
    if (!file.exists()) {
      file.createNewFile()
    }
    Files.write(
      Paths.get(filePath),
      (message + "\n").getBytes,
      StandardOpenOption.APPEND
    )
  }

  def time[R](block: => R, operation: String = "unknown", className: String = "UnknownClass", filePath: String = logFilePath): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    val elapsedMillis = (t1 - t0) / 1000000
    val elapsedNanos = t1 - t0
    val logMessage = s"[$className] Elapsed time for $operation:\t${elapsedMillis}ms (${elapsedNanos}ns)"
    logToFile(logMessage, filePath)
    println(logMessage)
    result
  }

  def roundAt(p: Int, n: Double): Double = {
    val s = math.pow(10, p)
    (math.round(n * s) / s)
  }
}
