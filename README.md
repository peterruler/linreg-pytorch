# linreg-pytorch

- Linear Regression with pytorch with numerical scalar target value

# demo

- URL: https://linreg-pytorch.onrender.com

# create a environment

```
conda create --name torch31111 python=3.11.11
```

# activate the environment

```
conda activate torch31111
```

# install dependencies

```
pip install -r requirements.txt
```

# install dependencies manually (optional)

```
conda install -y flask
conda install -y Flask-WTF
conda install -y joblib
conda install -y scikit-learn
pip install scikit-learn==1.6.1
conda install -y scipy
conda install -y werkzeug
conda install -y pandas
conda install -y numpy
conda install -y seaborn
conda install -y matplotlib
conda install -y pickle
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
conda install -y sklearn
```

# run flask server

```
python app.py
```

# call in a webbrowser

```
http://localhost:5000
```

# deploying the Application on Render.com

1. Fork this repository on GitHub.
2. Connect the GitHub repository to Render.com (sign in with your GitHub account).
3. Choose the free plan (0$) and the type "Web Service".
4. Build the application on Render.com:
```
pip install --upgrade pip && pip install -r requirements.txt
```

# set Start Command on Render.com

```
python app.py
```

# set Environment Variables on Render.com

```
PYTHON_VERSION => 3.11.11
PORT => 5000
```

# Explanation of the code (in Deutsch / German)

1. Importierte Bibliotheken

Datenverarbeitung und Visualisierung:

pandas, numpy zum Laden und Verarbeiten der Daten

matplotlib.pyplot und seaborn für grafische Darstellungen

Serialisierung:

pickle zum Speichern und Laden des Skalierers (MinMaxScaler)

PyTorch:

torch sowie Module wie nn, optim für den Aufbau und das Training des neuronalen Netzes

TensorDataset und DataLoader zur Erstellung von Mini-Batches

Sklearn:

train_test_split zur Aufteilung der Daten

MinMaxScaler zur Normalisierung der Eingabefeatures

Metriken wie mean_absolute_error und mean_squared_error zur Evaluierung

2. Datenvorbereitung und -vorverarbeitung

2.1. Google Drive Mounting & Laden der Daten

Drive Mounting:
Falls das Skript in Google Colab ausgeführt wird, wird das Google Drive gemountet, um auf die CSV-Datei zuzugreifen.

Daten laden:
Die CSV-Datei wird in ein Pandas DataFrame geladen:

```
df = pd.read_csv("/content/drive/My Drive/dl-udemy/fake_reg.csv")
```

Erste Datenanalyse:
Mit df.head() wird ein erster Blick auf die Daten geworfen, und mittels sns.pairplot(df) werden Zusammenhänge zwischen den Variablen visualisiert.

2.2. Aufteilen in Features und Zielvariable

Features:
Zwei Spalten (feature1, feature2) werden als Eingabefeatures verwendet.

Zielvariable:
Die Spalte price ist die Zielvariable.

Aufteilung:
Mit train_test_split wird das Dataset in Trainings- und Testdaten (70:30-Verhältnis) unterteilt.

2.3. Normalisierung der Features
MinMaxScaler:
Nur die Eingabefeatures werden skaliert (nicht die Zielvariable). Dadurch werden die Werte in einen Bereich zwischen 0 und 1 transformiert:

```
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

Speichern des Skalierers:
Der trainierte Scaler wird mit pickle gespeichert, sodass er später für Vorhersagen verwendet werden kann.

3. Umwandlung in PyTorch-Tensoren und Erstellung des DataLoaders

Tensoren:
Die numpy-Arrays der Trainings- und Testdaten werden in PyTorch-Tensoren konvertiert, wobei auch die Zielvariablen in 2D-Tensoren umgeformt werden.

DataLoader:
Ein TensorDataset wird erstellt und mit DataLoader in Mini-Batches (Batch-Größe = 32) eingeteilt, um das Training zu beschleunigen und stabiler zu gestalten.

4. Modellarchitektur in PyTorch

Das Modell wird als nn.Sequential definiert:

Eingabeschicht:
Zwei Eingabefeatures werden auf 4 Neuronen abgebildet.

Versteckte Schichten:
Es gibt drei versteckte Schichten mit jeweils 4 Neuronen und ReLU-Aktivierung.

Ausgabeschicht:
Eine einzelne Neuron-Schicht für die Regressionsvorhersage.

Beispiel:

```
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)
```

5. Definition von Verlustfunktion und Optimierer
Verlustfunktion:
Es wird der mittlere quadratische Fehler (MSELoss) verwendet.

Optimierer:
RMSprop wird als Optimierer mit einer erhöhten Lernrate von 0.01 verwendet, um schneller zu konvergieren:

```
optimizer = optim.RMSprop(model.parameters(), lr=0.01)
```

6. Training des Modells
Trainingsschleife:
Für 250 Epochen wird das Modell trainiert. Dabei werden:

Mini-Batches über den DataLoader iteriert.

Für jeden Batch wird:

Die Gradienten mit optimizer.zero_grad() zurückgesetzt.

Eine Vorwärtsausgabe berechnet.

Der Verlust berechnet und mittels loss.backward() der Backward-Pass durchgeführt.

Die Gewichte mittels optimizer.step() aktualisiert.

Verlustverlauf:
Der durchschnittliche Verlust jeder Epoche wird gespeichert und alle 50 Epochen ausgegeben.

7. Visualisierung und Evaluierung
Verlustplot:
Der Trainingsverlust wird über alle Epochen hinweg mit seaborn.lineplot visualisiert.

Modellbewertung:
Das Modell wird auf den Trainings- und Testdaten evaluiert, indem die MSE berechnet wird.

Vorhersagevergleich:
Die Vorhersagen auf den Testdaten werden mit den tatsächlichen Werten verglichen:

Ein Scatterplot zeigt die Übereinstimmung zwischen den Testwerten und den Modellvorhersagen.

Ein Histogramm der Fehlerverteilung wird erstellt.

Metriken:
Zusätzlich werden MAE, MSE und RMSE berechnet, um die Vorhersagegüte zu beurteilen.

8. Vorhersage für neue Daten
Neuer Datenpunkt:
Ein Beispielwert [[998, 1000]] wird als neuer Datenpunkt definiert.

Skalierung:
Dieser neue Datenpunkt wird mit dem zuvor gespeicherten MinMaxScaler skaliert.

Vorhersage:
Das Modell gibt für diesen skalierten Datenpunkt eine Vorhersage ab, die im ursprünglichen Maßstab (z. B. etwa 418.4327) interpretiert werden soll.

9. Speichern und Laden des Modells
Speichern:
Mit torch.save(model.state_dict(), 'model.pth') wird der aktuelle Zustand des Modells gespeichert.

Laden:
Ein identisches Modell wird neu definiert und mittels loaded_model.load_state_dict(torch.load('model.pth')) wiederhergestellt. Anschließend wird das geladene Modell getestet, um sicherzustellen, dass die Vorhersagen übereinstimmen.

Zusammenfassung
Dieses Skript illustriert den vollständigen Workflow eines Regressionsmodells in PyTorch:

Daten laden und visualisieren

Vorverarbeitung und Normalisierung der Features

Erstellen von Tensoren und DataLoadern für das Training

Definieren einer Modellarchitektur mit mehreren Schichten und Aktivierungsfunktionen

Training des Modells mittels Mini-Batch-Training und RMSprop-Optimierung

Evaluierung der Modellleistung durch Visualisierung und Fehlerberechnung

Durchführen von Vorhersagen auf neuen, skalierten Daten

Persistieren und Wiederherstellen des trainierten Modells
