# Sistema di Raccomandazione per Giochi Steam

Questo progetto implementa un sistema di raccomandazione basato su Scala e Spark, che utilizza **TF-IDF** e **Cosine Similarity** per generare raccomandazioni personalizzate. Il sistema sfrutta i tag dei giochi per suggerire titoli in linea con i gusti degli utenti, analizzando le librerie e identificando utenti con preferenze simili.

## Dataset

Il progetto utilizza un dataset pubblico di Kaggle contenente:
- Recensioni
- Giochi giocati
- Tag
- Metadati dei giochi

Il dataset è suddiviso in quattro file principali:
- `recommendations.csv`
- `games.csv`
- `games_metadata.json`
- `users.csv`

Il dataset completo pesa circa 3 GB.

## Obiettivi del Progetto

L'obiettivo principale è testare diverse implementazioni del codice in Scala e Spark, sia in locale che in cloud, per analizzare:
- Tempi di esecuzione
- Performance
- Scalabilità

Il progetto include l'utilizzo di un cluster **Google Cloud DataProc** per test su larga scala.

---

## Struttura del Progetto

Il sistema è suddiviso in versioni differenti che implementano approcci diversi per lo stesso scopo:

1. **Versione Sequenziale**
   - Utilizza Scala e le Scala Collections.
   - Limitata a dataset personalizzati di piccole dimensioni ed eseguibile solo in locale.

2. **Versione Parallela**
   - Utilizza Scala e le Scala Parallel Collections.
   - Migliora le performance rispetto alla versione sequenziale, ma con le stesse limitazioni.

3. **Versioni Distribuite con Spark**
   - **Spark SQL**
   - **Spark RDD**
   - **Spark MLLIB** (utilizzando librerie Spark per TF-IDF e Cosine Similarity)

Queste versioni supportano dataset di grandi dimensioni ed esecuzioni distribuite su cluster.

---

## Processo di Realizzazione

### 1. Preprocessing dei Dati

I dati vengono letti e uniti dai file sorgenti utilizzando diverse tecniche, come:
- **API Scala/Spark** per CSV e JSON.
- Filtraggio delle recensioni negative per mantenere solo raccomandazioni positive.

Il risultato è un dataset unificato, pronto per il calcolo delle raccomandazioni.

### 2. Calcolo del TF-IDF

Il **TF-IDF (Term Frequency-Inverse Document Frequency)** calcola l'importanza dei tag per ogni utente:
- **TF**: Frequenza di un tag nella lista dell'utente.
- **IDF**: Importanza globale del tag nel dataset.

L'output è una lista di punteggi TF-IDF per i tag associati a ciascun utente.

### 3. Calcolo della Cosine Similarity

La **Cosine Similarity** misura la similarità tra gli utenti:
- **Prodotto scalare**: Somma dei prodotti delle componenti corrispondenti dei vettori TF-IDF.
- **Magnitudine del vettore**: Radice quadrata della somma dei quadrati delle componenti.

L'output è una lista di utenti simili ordinati per similarità.

### 4. Generazione delle Raccomandazioni

Le raccomandazioni vengono generate recuperando i giochi dai top N utenti più simili, escludendo quelli già posseduti dall'utente target.

### 5. Esecuzione su Cluster Google Cloud DataProc

Dopo i test in locale, il codice è stato eseguito su un cluster Google Cloud DataProc per valutare scalabilità e performance.

**Configurazione del cluster:**
- Regione: `us-central1`
- Zona: `us-central1-f`
- Master Node: `n2-highmem-4` con SSD 80GB
- Worker Nodes: 7 `n2-highmem-4` con SSD 60GB

Comando per creare il cluster:
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
## Come scaricare ed eseguire

### Clonare il repository
Clona il [repository][github-repo-link] da GitHub nella posizione desiderata sul tuo file system locale.

```powershell
git clone https://github.com/samuele-lolli/recommendationsystem.git
```
Se richiesto, inserisci le tue credenziali GitHub e attendi il completamento del processo.

### Scaricare IntelliJ Idea
Scarica e installa l'IDE da [questo link][intellij].

### Scaricare i plugion
   Una volta che IntelliJ è installato, avvialo e clicca sull'ingranaggio nella barra degli strumenti in alto a destra. 
   Clicca su **Plugins**, quindi cerca e installa:
    - [**Scala**][scala-plugin-link]
    - [**Spark**][spark-plugin-link]
### Scarica il dataset
   Scarica il dataset da [kaggle][ds-link] ed estrailo dal file ```.zip```
   Nel progetto, crea una cartella ```steam-dataset``` e sposta al suo interno tutto il contenuto di ```archive.zip```

### Versioni Java e Scala
   Dopo aver scaricato entrambi i plugin, clicca ancora sulle impostazioni e seleziona **Platform Settings**
   Da questo menù, scegli **SDKs**, poi aggiungi un nuovo SDK dall'icona **+** in alto, e scarica **Java JDK version 18.0.2**. Quando il download termina, premi **Apply**
   Sempre in **Platform Settings**, scegli **Global Libraries → + → Scala SDK → Version 2.12.24 → OK → Apply**

### Run locale
   Se vuoi eseguire il progetto in locale, questo è tutto ciò che ti serve. Esegui la classe main dalla barra delle run configurations nella zona in alto a destra della finestra dell'IDE

### Cloud run
1. Scarica ed installa [```gcloud CLI``` CLI][gcloud-link]
2. Esegui  ```sbt assembly``` da terminale nella cartella del progetto per creare un file JAR. Puoi configurare nome e destinazione di quest'ultimo modificando ```build.sbt```
3. Una volta fatto, esegui ```gcloud dataproc jobs submit spark``` con gli argomenti necessari

Esempio:
```
gcloud dataproc jobs submit spark
--cluster cluster-0000
--region us-central1
--class com.example.class
--jars C:\path\to\jar\recommendationSystem.jar
```

---

## Problemi Incontrati e Soluzioni

### Persistenza/Cache
Alcune tecniche di persistenza si sono rivelate inefficaci in cloud. Sono stati effettuati test per ottimizzare le strategie in base all'ambiente di esecuzione.

### Cambiamento di Approccio
Inizialmente, il sistema utilizzava i titoli dei giochi anziché i tag. Questo approccio si è rivelato meno preciso, portando alla decisione di basarsi sui tag.

### Dataset Personalizzati
Per le versioni sequenziale e parallela, sono stati generati dataset ridotti per supportare test locali più efficienti.

### Allineamento Output
È stato necessario allineare i risultati delle diverse implementazioni per garantire coerenza tra approcci (RDD, SQL, MLLIB).

### Misurazione dei Tempi
La natura lazy di Spark ha complicato la misurazione accurata dei tempi di esecuzione. Sono state adottate tecniche per migliorare la precisione delle stime.

---

## Conclusioni

Il progetto dimostra l'utilizzo di tecniche sia locali che distribuite per generare raccomandazioni personalizzate. Offre un'analisi delle differenze tra:
- Approcci sequenziali, paralleli e distribuiti
- Scalabilità e performance in ambienti cloud

Il progetto è stato realizzato per l'esame di **Scalable and Cloud Programming** all'Università di Bologna (Anno Accademico 2023-2024) da [Leonardo Vincenzi][leonardo], [Samuele Lolli][samuele] e [Giulio Becchi][giulio].

[leonardo]: https://github.com/leonardovincenzi1998
[samuele]: https://github.com/samuele-lolli
[giulio]: https://github.com/gbekss

