# Steam Game Recommendation System

This project implements a recommendation system based on Scala and Spark, using **TF-IDF** and **Cosine Similarity** to generate personalized recommendations. The system leverages the following dataset from Kaggle:

## Dataset

The project uses a public dataset from Kaggle containing:
- Reviews
- Played games
- Tags
- Game metadata

The dataset is divided into four main files:
- `recommendations.csv`
- `games.csv`
- `games_metadata.json`
- `users.csv`

The complete dataset weighs about 3 GB.

## Project Objectives

The main objective is to test different implementations of the code in Scala and Spark, both locally and in the cloud, to analyze:
- Execution times
- Performance
- Scalability

The project includes the use of a **Google Cloud DataProc** cluster for large-scale tests.

## Project Structure

The system is divided into different versions that implement different approaches for the same purpose:

1. **Sequential Version**
   - Uses Scala and Scala Collections.
   - Limited to small custom datasets and executable only locally.

2. **Parallel Version**
   - Uses Scala and Scala Parallel Collections.
   - Improves performance compared to the sequential version, but with the same limitations.

3. **Distributed Versions with Spark**
   - **Spark SQL**
   - **Spark RDD**
   - **Spark MLLIB** (using Spark libraries for TF-IDF and Cosine Similarity)

These versions support large datasets and distributed executions on clusters.

## Implementation Process

### 1. Data Preprocessing

The data is read and merged from the source files using various techniques, such as:
- **Scala/Spark API** for CSV and JSON.
- Filtering out negative reviews to keep only positive recommendations.

The result is a unified dataset, ready for recommendation calculations.

### 2. TF-IDF Calculation

**TF-IDF (Term Frequency-Inverse Document Frequency)** calculates the importance of tags for each user:
- **TF**: Frequency of a tag in the user's list.
- **IDF**: Global importance of the tag in the dataset.

The output is a list of TF-IDF scores for the tags associated with each user.

### 3. Cosine Similarity Calculation

**Cosine Similarity** measures the similarity between users:
- **Dot product**: Sum of the products of the corresponding components of the TF-IDF vectors.
- **Vector magnitude**: Square root of the sum of the squares of the components.

The output is a list of similar users ordered by similarity.

### 4. Recommendation Generation

Recommendations are generated by retrieving games from the top N most similar users, excluding those already owned by the target user.

### 5. Execution on Google Cloud DataProc Cluster

After local tests, the code was run on a Google Cloud DataProc cluster to evaluate scalability and performance.

**Cluster Configuration:**
- Region: `us-central1`
- Zone: `us-central1-f`
- Master Node: `n2-highmem-4` with 80GB SSD
- Worker Nodes: 7 `n2-highmem-4` with 60GB SSD

Command to create the cluster:
```bash
gcloud dataproc clusters create cluster-7w \
  --enable-component-gateway \
  --region us-central1 \
  --zone us-central1-f \
  --master-machine-type n2-highmem-4 \
  --master-boot-disk-type pd-ssd \
  --master-boot-disk-size 80 \
  --num-master-local-ssds 1 \
  --num-workers 7 \
  --worker-machine-type n2-highmem-4 \
  --worker-boot-disk-type pd-ssd \
  --worker-boot-disk-size 60 \
  --num-worker-local-ssds 2 \
  --image-version 2.1-debian11 \
  --scopes 'https://www.googleapis.com/auth/cloud-platform' \
  --project recommendation2324
```

## How to Download and Run

### Clone the repository
Clone the [repository](https://github.com/samuele-lolli/recommendationsystem.git) from GitHub to your desired location on your local file system.
```powershell
git clone https://github.com/samuele-lolli/recommendationsystem.git
```

### Download IntelliJ Idea
Download and install the IDE from [this link](https://www.jetbrains.com/idea/download/).

### Download Plugins
Once IntelliJ is installed, launch it and click on the gear icon in the top right toolbar. Click on **Plugins**, then search for and install:
* [**Scala**](https://plugins.jetbrains.com/plugin/1347-scala)
* [**Spark**](https://plugins.jetbrains.com/plugin/21700-spark)

### Download the dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) and extract it from the `.zip` file. In the project, create a folder `steam-dataset` and move all the contents of `archive.zip` into it.

### Java and Scala Versions
After downloading both plugins, click on the settings and select **Platform Settings**. From this menu, choose **SDKs**, then add a new SDK by clicking the **+** icon at the top and download **Java JDK version 18.0.2**. When the download is complete, press **Apply**. In **Platform Settings**, choose **Global Libraries → + → Scala SDK → Version 2.12.24 → OK → Apply**.

### Local Run
If you want to run the project locally, this is all you need. Run the main class from the run configurations bar at the top right of the IDE window.

### Cloud Run
1. Download and install [```gcloud CLI```](https://cloud.google.com/sdk/docs/install?hl=it).
2. Run ```sbt assembly``` from the terminal in the project folder to create a JAR file. You can configure the name and destination of this file by modifying ```build.sbt```.
3. Once done, run ```gcloud dataproc jobs submit spark``` with the necessary arguments.

Example:
```bash
gcloud dataproc jobs submit spark \
--cluster cluster-0000 \
--region us-central1 \
--class com.example.class \
--jars C:\path\to\jar\recommendationSystem.jar
```

[leonardo]: https://github.com/leonardovincenzi1998
[samuele]: https://github.com/samuele-lolli
[giulio]: https://github.com/gbekss
[gcloud-link]: https://cloud.google.com/sdk/docs/install?hl=it
[github-repo-link]: https://github.com/samuele-lolli/recommendationsystem.git
[intellij]: https://www.jetbrains.com/idea/download/
[scala-plugin-link]: https://plugins.jetbrains.com/plugin/1347-scala
[spark-plugin-link]: https://plugins.jetbrains.com/plugin/21700-spark
[ds-link]: https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam
