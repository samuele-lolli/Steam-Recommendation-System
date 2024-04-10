# Steam Recommendation System

## How to download and run

### Clone the repository
Clone the [repository][github-repo-link] from GitHub at the desired location in your local file system.

```powershell
git clone https://github.com/samuele-lolli/recommendationsystem.git
```
If requested, submit your GitHub credentials and wait for the process to finish

### Download IntelliJ Idea
Download and install the IDE from [this link][intellij]

### Download relevant plugins
1. **Plugins**  
   Once IntelliJ is installed, launch it and click on the cogweel in the top-right toolbar.
   Click on **Plugins**, then search and install.
    - [**Scala**][scala-plugin-link]
    - [**Spark**][spark-plugin-link]
2. **Dataset download**  
   Download the dataset from [kaggle][ds-link] and extract it from the downloaded zip file.
   Within the project, create a new folder named ```steam-dataset``` and paste there all the contents of ```archive.zip```
2. **Java and Scala versions**  
   After the download of both plugins, click again on the cogwheel in the top right and select **Platform Settings**.
   From this menu, choose **SDKs**, then add a new SDK via the "plus" icon at the top of the window, and dowload Java JDK version 18.0.2. Once done, hit "Apply"
   Then, always in the **Platform Settings** menu, choose **Global Libraries -> + -> Scala SDK -> Version 2.12.24 -> OK -> Apply**
3. **Local Run**  
   If you want to run the project locally, you're good to go! Just select *current file* option in the run configurations toolbar in the top right of the IDE window.
4. **Cloud Run**  
   Run ```sbt assembly``` in the project folder's terminal to create a JAR file. You can configure the jar name and destination root by editing ```build.sbt```.
   If you have a Google Cloud account, you can also connect the project to a Spark Cluster by creating a specific run configuration. Follow the steps to authenticate with your Google account and in the run configurations popup select the desired cluster and the ```.jar``` file you created earlier.
   

### Development
This project was created for the Scalable and Cloud programming exam at Bologna University, Master's Degree in Information Technology (Academic year 2023-2024) by [Vincenzi Leonardo][leonardo], [Lolli Samuele][samuele] and [Giulio Becchi][giulio].
   


[github-repo-link]: https://github.com/samuele-lolli/recommendationsystem
[intellij]: https://www.jetbrains.com/idea/download/?section=windows
[scala-plugin-link]: https://plugins.jetbrains.com/plugin/1347-scala
[spark-plugin-link]: https://plugins.jetbrains.com/plugin/21700-spark
[ds-link]: https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam
[leonardo]: https://github.com/leonardovincenzi1998
[samuele]: https://github.com/samuele-lolli
[giulio]: https://github.com/gbekss
