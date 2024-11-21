import sbtassembly.AssemblyPlugin.autoImport.assembly

ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.14"

enablePlugins(sbtassembly.AssemblyPlugin)



lazy val root = (project in file("."))
  .settings(
    name := "recommendationsystem",

    assembly / mainClass := Some("com.unibo.recommendationsystem.distributedMain"),

    // Configure assembly settings
    assembly / assemblyJarName := "recommendationSystem.jar",
    assemblyMergeStrategy := {
      case PathList("META-INF", "MANIFEST.MF") => MergeStrategy.discard
      case PathList("META-INF", _*) => MergeStrategy.discard
      case "reference.conf" => MergeStrategy.concat
      case _ => MergeStrategy.first
    }
  )

val sparkVersion = "3.3.2"

libraryDependencies += "com.typesafe.play" %% "play-json" % "2.9.2"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.scala-lang" % "scala-library" % "2.12.14",
  "com.google.cloud" % "google-cloud-storage" % "2.40.1"
)