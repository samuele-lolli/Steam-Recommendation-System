package com.unibo.recommendationsystem.utils

import java.io.{FileWriter, PrintWriter}

object timeUtils {
  /** Path to the log file where messages will be saved. */
  private var logFilePath = "result.txt"

  /**
   * Sets the path for the log file.
   *
   * @param path String representing the new log file path.
   */
  def setLogFilePath(path: String): Unit = {
    logFilePath = path
  }

  /**
   * Logs a message to the specified log file.
   * The log is appended to the file.
   *
   * @param message The message to be logged.
   */
  private def logToFile(message: String): Unit = {
    val writer = new PrintWriter(new FileWriter(logFilePath, true))
    try {
      writer.println(message)
    } finally {
      writer.close()
    }
  }

  /**
   * Saves a user-provided input message to the log file, prefixed with "Input:".
   *
   * @param message String containing the user input to save.
   */
  def saveUserInput(message: String): Unit = {
    logToFile("Input: " + message)
  }

  /**
   * Measures the execution time of a code block and logs the result.
   * Logs the elapsed time in milliseconds and seconds, along with the operation and class name.
   *
   * @param block The block of code to time.
   * @param operation A description of the operation being measured (default is "unknown").
   * @param className The name of the class or context in which the operation occurs (default is "UnknownClass").
   * @tparam R The return type of the code block.
   * @return The result of the executed block.
   */
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
