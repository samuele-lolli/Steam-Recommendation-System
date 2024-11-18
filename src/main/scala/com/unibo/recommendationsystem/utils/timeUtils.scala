package com.unibo.recommendationsystem.utils

import org.sparkproject.dmg.pmml.True

import java.io.{FileWriter, PrintWriter}

object timeUtils {

  private var logFilePath = "result.txt"

  def setLogFilePath(path : String): Unit = {
    logFilePath = path
  }

  private def logToFile(message: String): Unit = {
    val writer = new PrintWriter(new FileWriter(logFilePath, true))
    try {
      writer.println(message)
    } finally {
      writer.close()
    }
  }

  def saveUserInput(message: String) : Unit = {
      logToFile("Input: " + message)
  }


  def time[R](block: => R, operation: String = "unknown", className: String = "UnknownClass"): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()

    val elapsedMillis = (t1 - t0) / 1000000
    val elapsedSeconds = (t1 - t0) / 1e9

    val logMessage = f"[$className] Elapsed time for $operation: $elapsedMillis%d ms ($elapsedSeconds%.3f s)"
    logToFile(logMessage)
    println(logMessage)

    result
  }
}
