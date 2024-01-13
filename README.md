# Detectarea Tumorilor Cerebrale folosind KNN și Naive Bayes

## Descriere Proiect

Acest proiect abordează detectarea tumorilor cerebrale prin analiza imaginilor de rezonanță magnetică (MRI) utilizând algoritmii K-Nearest Neighbors (KNN) și Naive Bayes (NB). În plus, rezultatele acestor metode sunt comparate cu cele obținute prin algoritmul Support Vector Machine (SVM) pentru a evalua eficacitatea diferitelor abordări.

## Codul Sursă

- `knn_nb_stack.py`: Implementarea algoritmului de stacking care combină KNN și NB cu Logistic Regression ca meta-învățător.
- `svm_model.py`: Codul sursă pentru implementarea modelului SVM (furnizat separat).

## Procesul de Antrenare și Evaluare

Modelul KNN/NB a fost antrenat pe un set de date constând din imagini MRI preprocesate și etichetate. Performanța modelului a fost evaluată în funcție de acuratețe, precizie, recall și scorul F1. Similar, modelul SVM a fost evaluat folosind aceleași metrici pentru a permite o comparație directă.

## Rezultate și Comparare

- **Rezultate KNN/NB**: Detalii despre acuratețea, precizia, recall-ul și scorul F1 obținut de modelul KNN/NB.
- **Rezultate SVM**: Informații similare despre performanța modelului SVM.
- **Comparare**: O analiză a performanțelor celor două abordări, evidențiind avantajele și limitările fiecăreia.

## Utilizare

Pentru a rula modelele și a evalua performanța acestora:

```bash
python knn_nb_stack.py
python svm_model.py

