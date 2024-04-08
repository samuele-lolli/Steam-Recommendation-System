import sbtassembly.AssemblyPlugin.autoImport.assembly

ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.14"

enablePlugins(sbtassembly.AssemblyPlugin)


lazy val root = (project in file("."))
  .settings(
    name := "recommendationsystem",
    assemblyMergeStrategy := {
      case PathList("META-INF", _*) => MergeStrategy.discard
      case x => MergeStrategy.first
    },
    assembly / assemblyJarName := "recommendationMLlib.jar"
  )


val sparkVersion = "3.3.2"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.scala-lang" % "scala-library" % "2.12.14"
)
