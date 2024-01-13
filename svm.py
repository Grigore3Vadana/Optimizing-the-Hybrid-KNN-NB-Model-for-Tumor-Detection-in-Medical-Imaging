# Importăm modulele necesare pentru lucrul cu fișiere, calcul numeric, prelucrarea imaginilor și algoritmi de machine learning.
import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, auc, roc_curve, \
    precision_recall_curve, average_precision_score

# Definim o funcție care va încărca o imagine dintr-o cale dată, o va converti în tonuri de gri și o va redimensiona.
def load_and_preprocess_image(image_path):
    image = imread(image_path)  # Citim imaginea din calea specificată
    image = rgb2gray(image)  # Convertim imaginea în tonuri de gri pentru a reduce complexitatea
    image = resize(image, (256, 256))  # Redimensionăm imaginea la dimensiunea dorită
    return image / 255.0  # Normalizăm valorile pixelilor la o scară de 0 la 1

# Definim o funcție pentru a încărca toate imaginile și etichetele lor dintr-un anumit folder.
def load_images_from_folder(folder, label):
    images = []  # Inițializăm o listă pentru a stoca imaginile
    labels = []  # Inițializăm o listă pentru a stoca etichetele
    for filename in os.listdir(folder):  # Iterăm prin fișierele din director
        if filename.endswith('.jpg'):  # Verificăm dacă fișierul este o imagine
            img = load_and_preprocess_image(os.path.join(folder, filename))  # Preprocesăm fiecare imagine
            images.append(img.flatten())  # Aplatizăm imaginea într-un vector unidimensional și o adăugăm la listă
            labels.append(label)  # Adăugăm eticheta corespunzătoare în lista de etichete
    return images, labels  # Returnăm listele cu imagini și etichete

# Încărcăm imaginile cu tumori și fără tumori, folosind calea și eticheta corespunzătoare.
tumor_images, tumor_labels = load_images_from_folder(r'C:\Users\Grig\PycharmProjects\ML\Data\tumor', 1)
no_tumor_images, no_tumor_labels = load_images_from_folder(r'C:\Users\Grig\PycharmProjects\ML\Data\no_tumor', 0)

# Combinăm imaginile și etichetele într-un singur set de date pentru a fi folosit în antrenarea modelului.
data = tumor_images + no_tumor_images  # Combinăm imaginile într-o singură listă
labels = tumor_labels + no_tumor_labels  # Combinăm etichetele într-o singură listă

# Împărțim setul de date în subseturi pentru antrenament, validare și testare.
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)  # Împărțirea inițială
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Împărțirea suplimentară pentru validare și test

# Standardizăm datele pentru a îmbunătăți performanța și stabilitatea numerică a algoritmului SVM.
scaler = StandardScaler()  # Inițializăm un obiect StandardScaler
X_train_scaled = scaler.fit_transform(X_train)  # Antrenăm scaler-ul și transformăm setul de antrenament
X_val_scaled = scaler.transform(X_val)  # Transformăm setul de validare
X_test_scaled = scaler.transform(X_test)  # Transformăm setul de test

# Antrenăm un model SVM cu un kernel liniar folosind setul de antrenament.
svm = SVC(kernel='linear')  # Inițializăm modelul SVM cu kernel liniar
svm.fit(X_train_scaled, y_train)  # Antrenăm modelul cu datele standardizate

# Evaluăm modelul pe setul de validare și afișăm rezultatele.
y_val_pred = svm.predict(X_val_scaled)  # Facem predicții pe setul de validare
print("Validation Results:")  # Afișăm rezultatele validării
print(classification_report(y_val, y_val_pred))  # Afișăm raportul de clasificare pentru setul de validare

# Evaluăm modelul pe setul de test pentru a verifica generalizabilitatea sa.
y_test_pred = svm.predict(X_test_scaled)  # Facem predicții pe setul de test
print("Test Results:")  # Afișăm rezultatele testării
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))  # Afișăm acuratețea pe setul de test
print(classification_report(y_test, y_test_pred))  # Afișăm raportul de clasificare pentru setul de test

# Utilizăm funcții de vizualizare pentru a interpreta mai ușor performanța modelului.
import matplotlib.pyplot as plt
import seaborn as sns

# Afișăm o matrice de confuzie pentru a vizualiza erorile de predicție ale modelului.
cm = confusion_matrix(y_test, y_test_pred)  # Calculăm matricea de confuzie
plt.figure(figsize=(10,7))  # Setăm dimensiunea figurii
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Creăm un heatmap pentru matricea de confuzie
plt.xlabel('Predicted')  # Etichetăm axa x
plt.ylabel('Truth')  # Etichetăm axa y
plt.title('Confusion Matrix')  # Dăm un titlu figurii
plt.show()  # Afișăm figura

# Calculăm și afișăm curba ROC pentru a evalua performanța modelului la diferite praguri de clasificare.
fpr, tpr, _ = roc_curve(y_test, svm.decision_function(X_test_scaled))  # Calculăm valorile pentru curba ROC
roc_auc = auc(fpr, tpr)  # Calculăm aria de sub curba ROC

plt.figure()  # Inițiem o nouă figură
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)  # Desenăm curba ROC
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Desenăm linia de bază
plt.xlabel('False Positive Rate')  # Etichetăm axa x
plt.ylabel('True Positive Rate')  # Etichetăm axa y
plt.title('Receiver Operating Characteristic')  # Dăm un titlu figurii
plt.legend(loc="lower right")  # Adăugăm legenda
plt.show()  # Afișăm figura

# Calculăm și afișăm curba Precision-Recall, care este utilă pentru evaluarea performanței în caz de dezechilibru de clase.
precision, recall, _ = precision_recall_curve(y_test, svm.decision_function(X_test_scaled))  # Calculăm valorile pentru curba Precision-Recall
average_precision = average_precision_score(y_test, svm.decision_function(X_test_scaled))  # Calculăm scorul mediu de precizie

plt.figure()  # Inițiem o nouă figură
plt.step(recall, precision, where='post', label='Average precision score, AP={0:0.2f}'.format(average_precision))  # Desenăm curba Precision-Recall
plt.xlabel('Recall')  # Etichetăm axa x
plt.ylabel('Precision')  # Etichetăm axa y
plt.ylim([0.0, 1.05])  # Setăm limitele pentru axa y
plt.xlim([0.0, 1.0])  # Setăm limitele pentru axa x
plt.title('2-class Precision-Recall curve')  # Dăm un titlu figurii
plt.legend(loc="upper right")  # Adăugăm legenda
plt.show()  # Afișăm figura

# Definim o funcție pentru a afișa imagini împreună cu eticheta prezisă de model.
def display_image_with_label(image, label):
    plt.imshow(image.reshape(256, 256), cmap='gray')  # Afișăm imaginea
    plt.title(f'Predicted Label: {"Tumor" if label == 1 else "No Tumor"}')  # Setăm titlul cu eticheta prezisă
    plt.axis('off')  # Dezactivăm axele
    plt.show()  # Afișăm figura

# Iterăm prin imagini și etichetele lor reale pentru a afișa rezultatele modelului și pentru a le compara cu adevărul de teren.
for i in range(len(X_test_scaled)):
    image = X_test_scaled[i]  # Selectăm imaginea din setul de testare
    true_label = y_test[i]  # Selectăm eticheta reală
    predicted_label = svm.predict([image])[0]  # Facem predicția modelului

    # Redimensionăm imaginea pentru a inversa transformarea standardizării și pentru a o afișa
    image_reshaped = image.reshape(1, -1)  # Redimensionăm imaginea pentru a se potrivi cu scaler-ul
    original_image = scaler.inverse_transform(image_reshaped).reshape(256, 256)  # Inversăm scalarea și redimensionăm imaginea
    display_image_with_label(original_image, predicted_label)  # Afișăm imaginea cu eticheta prezisă

    # Opțional, afișăm eticheta reală pentru a compara cu predicția modelului
    print(f"True Label: {'Tumor' if true_label == 1 else 'No Tumor'}")
