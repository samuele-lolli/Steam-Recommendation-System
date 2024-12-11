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
   Then, always in the **Platform Settings** menu, choose **Global Libraries → + → Scala SDK → Version 2.12.24 → OK → Apply**
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


# Sistema di Raccomandazione per Giochi Steam
Questo progetto implementa un sistema di raccomandazione basato su Scala e Spark che utilizza **TF-IDF** e **Cosine Similarity** per generare raccomandazioni di giochi personalizzate. Il sistema sfrutta i tag dei giochi per suggerire titoli adatti ai gusti di un utente, identificando altri utenti con gusti simili ed estraendone la libreria.

## Dataset
Per il progetto è stato utilizzato un dataset pubblico di Kaggle contenente informazioni relative a Steam, tra cui:
- Recensioni
- Giochi giocati
- Tag
- Metadati dei giochi

Il dataset è suddiviso in quattro file principali:
- `recommendations.csv`
- `games.csv`
- `games_metadata.json`
- `users.csv`

Il dataset totale pesa circa 3 GB.

## Obiettivi del Progetto
L'obiettivo del progetto è testare diverse versioni di codice scritto in Scala e Spark, sia in locale che in cloud, verificando:
- Tempi di esecuzione
- Performance
- Scalabilità

Il progetto include l'utilizzo di un cluster **Google Cloud DataProc**, la cui configurazione viene dettagliata in seguito.

---

## Struttura del Progetto
Il sistema è suddiviso in diverse versioni del codice che implementano approcci differenti per svolgere lo stesso compito:

1. **Versione Sequenziale**
   - Utilizza Scala e le Scala Collections.
   - Funziona solo con dataset personalizzati di piccole dimensioni e in locale.

2. **Versione Parallela**
   - Utilizza Scala e le Scala Parallel Collections.
   - Migliora le performance rispetto alla versione sequenziale, ma ha le stesse limitazioni.

3. **Versioni Distribuite con Spark**
   - **Spark SQL**
   - **Spark RDD**
   - **Spark MLLIB** (utilizzando librerie Spark per TF-IDF e Cosine Similarity)

Queste versioni supportano l'elaborazione di grandi dataset e sono eseguibili su cluster distribuiti su Google Cloud DataProc.

---

## Processo di Realizzazione

### 1. Preprocessing dei Dati
Il preprocessing prepara i dati per il calcolo di TF-IDF e Cosine Similarity. I dati vengono letti e uniti dai file sorgenti attraverso diverse tecniche, tra cui:
- **Scala/Spark API** per CSV e JSON.
- Filtraggio delle recensioni negative per mantenere solo raccomandazioni positive.

Il risultato è un dataset unificato contenente tutte le informazioni necessarie per le raccomandazioni.

### 2. Calcolo del TF-IDF
Il **TF-IDF (Term Frequency-Inverse Document Frequency)** valuta l'importanza dei tag per ogni utente:
- **TF**: Frequenza di un tag nella lista dei tag di un utente.
- **IDF**: Importanza globale del tag nel dataset.

Il punteggio TF-IDF per un tag viene calcolato moltiplicando TF per IDF. L'output è una lista di punteggi associati ai tag della libreria di ogni utente.

### 3. Calcolo della Cosine Similarity
La **Cosine Similarity** misura la similarità tra utenti:
- **Prodotto scalare**: Somma dei prodotti delle componenti corrispondenti dei vettori TF-IDF.
- **Magnitudine del vettore**: Radice quadrata della somma dei quadrati delle componenti.

L'output è una lista di utenti simili all'utente target, ordinati per punteggio di similarità.

### 4. Generazione delle Raccomandazioni
Le raccomandazioni finali vengono generate recuperando i giochi dei top N utenti più simili, escludendo i giochi già posseduti dall'utente target.

### 5. Esecuzione su Cluster Google Cloud DataProc
Dopo i test in locale, il codice è stato eseguito su un cluster DataProc per testare scalabilità e performance.

Configurazione cluster:
- Regione: `us-central1`
- Zona: `us-central1-f`
- Master Node: `n2-highmem-4` con SSD 80GB
- Worker Nodes: 7 `n2-highmem-4` con SSD 60GB

Comando per la creazione del cluster:
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

---

## Problemi Incontrati e Soluzioni

### Persistenza/Cache
In locale, alcune tecniche di persistenza si sono rivelate inefficaci in cloud. Sono stati effettuati test per identificare le soluzioni migliori in base all'ambiente di esecuzione.

### Approccio Alternativo
Inizialmente, il progetto si basava sui titoli dei giochi anziché sui tag. Tuttavia, questo approccio forniva raccomandazioni meno precise, portando al cambio di strategia verso l'uso dei tag.

### Dataset Personalizzati
Per le versioni parallela e sequenziale, è stata implementata una funzione per generare dataset ridotti, mantenendo solo gli utenti target.

### Allineamento Output
Diverse implementazioni hanno richiesto un allineamento degli output per garantire risultati coerenti, indipendentemente dall'approccio (RDD, SQL, MLLIB).

### Misurazione Tempi di Esecuzione
La natura lazy di Spark ha complicato la misurazione accurata dei tempi. Sono state adottate soluzioni per ottenere stime più affidabili.

---

## Conclusioni
Il progetto dimostra come utilizzare tecniche distribuite e locali per generare raccomandazioni personalizzate. Offre spunti utili su:
- Differenze tra approcci sequenziali, paralleli e distribuiti.
- Scalabilità e performance in ambienti cloud.
