package com.unibo.recommendationsystem.utils

import java.io.{FileWriter, PrintWriter}

object timeUtils {
  /** Path to the log file where messages will be saved. */
  private var logFilePath = "result.txt"

  /**
   * Sets the path for the log file where messages will be logged.
   *
   * @param path The new log file path as a string.
   */
  def setLogFilePath(path: String): Unit = {
    logFilePath = path
  }

  /**
   * Logs a message to the specified log file by appending the message.
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
   * Saves a user-provided input message to the log file".
   *
   * @param message The user input message to save.
   */
  def saveUserInput(message: String): Unit = {
    logToFile(message)
  }

  /**
   * Measures the execution time of a code block and logs the result.
   * Logs the elapsed time in milliseconds and seconds, along with the operation and class name.
   *
   * @param block The block of code to be timed.
   * @param operation A description of the operation being measured (default is "unknown").
   * @param className The name of the class or context where the operation takes place (default is "UnknownClass").
   * @tparam R The return type of the code block.
   * @return The result of the executed code block.
   */
  def time[R](block: => R, operation: String = "unknown", className: String = "UnknownClass"): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()

    // Calculate the elapsed time in both milliseconds and seconds.
    val elapsedMillis = (t1 - t0) / 1000000
    val elapsedSeconds = (t1 - t0) / 1e9

    // Log the time taken for the operation, including the class name and operation description.
    val logMessage = f"[$className] Elapsed time for $operation: $elapsedMillis%d ms ($elapsedSeconds%.3f s)"
    logToFile(logMessage)
    println(logMessage)

    result
  }
}
