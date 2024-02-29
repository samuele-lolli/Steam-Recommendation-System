ThisBuild / version := "1.0.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.12"

lazy val root = (project in file("."))
  .settings(
    name := "untitled"
  )
//libraryDependencies += "com.fasterxml.jackson.databind" % "jackson-databind" % "2.13.0"


val sparkVersion = "3.5.0"

/*artifactName := { (sv: ScalaVersion, module: ModuleID, artifact: Artifact) =>
  "recommendation_system.jar" }*/

//libraryDependencies += "com.google.cloud.spark" %% "spark-bigquery" % "0.29.0"


libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.scala-lang" % "scala-library" % "2.13.12"
)