# Capitulo 3. Clasificación

# Proyecto de clasificación, para diagnosticar medicamentos basándose en las características de los pacientes.

# 1. Descargar los datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/a2Proyectos/MachineLearning_Data/main/"
LONDON_SALARY = "Capitulo_3/drug200.csv"
def extraer_datos(url, basedeDatos):
    csv_path = url + basedeDatos
    return pd.read_csv(csv_path)

df_drugs = extraer_datos(DOWNLOAD_ROOT, LONDON_SALARY)

# Visualiza el DataFrame
df_drugs.info()
print(df_drugs.head())

# 2. Análisis de cada Variable
print("Max Age:", df_drugs.Age.max())
print("Min Age:", df_drugs.Age.min())
plt.figure(figsize=(9,5))
sns.displot(df_drugs.Age, kde=True)
plt.show()

print(df_drugs.Sex.value_counts())

plt.figure(figsize=(9,5))
sns.histplot(data=df_drugs, x="BP", hue="BP")
plt.show()

plt.figure(figsize=(9,5))
sns.histplot(data=df_drugs, x="Cholesterol", hue="Cholesterol")
plt.show()

plt.figure(figsize=(9,5))
sns.displot(df_drugs.Na_to_K, kde=True)
plt.show()

plt.figure(figsize=(9,5))
sns.histplot(data=df_drugs, x="Drug", hue="Drug")
plt.show()
print(df_drugs.Drug.value_counts())

# 3. Análisis de Relación entre Variables
plt.figure(figsize=(9,5))
palette = sns.color_palette("Set2", n_colors=len(df_drugs.Drug.value_counts().index))
sns.swarmplot(x="Drug", y="Age", data=df_drugs, palette=palette)
plt.legend(df_drugs.Drug.value_counts().index)
plt.title("Edad/Medicamento")
plt.show()

df_Sex_Drug = df_drugs.groupby(["Drug","Sex"]).size().reset_index(name='Count')
print(df_Sex_Drug)
plt.figure(figsize=(9,5))
sns.barplot(x="Drug", y="Count", hue="Sex", data=df_Sex_Drug)
plt.title("Género/Medicamento")
plt.show()

df_BP_Drug = df_drugs.groupby(["Drug","BP"]).size().reset_index(name="Count")
plt.figure(figsize=(9,5))
sns.barplot(x="Drug", y="Count", hue="BP", data=df_BP_Drug)
plt.title("Presión Sanguinea/Medicamentos")
plt.show()

df_CH_Drug = df_drugs.groupby(["Drug","Cholesterol"]).size().reset_index(name="Count")
print(df_CH_Drug)
plt.figure(figsize=(9,5))
sns.barplot(x="Drug", y="Count", hue="Cholesterol", data=df_CH_Drug)
plt.title("Cholesterol -- Drug")
plt.show()

plt.figure(figsize=(9,5))
palette = sns.color_palette("Set2", n_colors=len(df_drugs.Drug.value_counts().index))
sns.swarmplot(x="Drug", y="Na_to_K", data=df_drugs, palette=palette)
plt.title("Sodio-Potasio/Medicamentos")
plt.show()

# 4. Limpieza y Separación de Datos
from sklearn.preprocessing import LabelEncoder
df = df_drugs.copy()
def label_encoder(datos_categoria):
    le = LabelEncoder()
    df_drugs[datos_categoria] = le.fit_transform(df_drugs[datos_categoria])
variables = ["Sex","BP","Cholesterol","Na_to_K","Drug"]
for l in variables:
    label_encoder(l)
print(df_drugs.head())

x = df_drugs.drop(["Drug"], axis=1)
y = df_drugs["Drug"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

# 5. Modelo de Clasificación Binario
y_train_c = (y_train == 0)
y_test_c = (y_test == 0)
print(y_train_c.value_counts())

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(random_state=42)
sgd.fit(x_train, y_train_c)

print(sgd.predict([[47,1,1,0,8]]))

# 6. Medidas de desempeño
from sklearn.model_selection import cross_val_score, cross_val_predict
print(cross_val_score(sgd, x_train, y_train_c, cv=3, scoring="accuracy"))

y_train_pred = cross_val_predict(sgd, x_train, y_train_c, cv=3)
print(y_train_pred)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
print(confusion_matrix(y_train_c, y_train_pred))

p = precision_score(y_train_c, y_train_pred)
r = recall_score(y_train_c, y_train_pred)
print("Precisión:", p, "Recall:", r)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=42)
rfc.fit(x_train, y_train_c)
y_train_pred_rf = cross_val_predict(rfc, x_train, y_train_c, cv=3)
print(confusion_matrix(y_train_c, y_train_pred_rf))
p_rf = precision_score(y_train_c, y_train_pred_rf)
r_rf = recall_score(y_train_c, y_train_pred_rf)
print("Precisión RF:", p_rf, "Recall RF:", r_rf)
print("F1 RF:", f1_score(y_train_c, y_train_pred_rf))

# Umbral Precision y Recall
y_score = sgd.decision_function([[47,1,1,0,8]])
print("Score paciente ejemplo:", y_score)
threshold = 0
print("Predicción con threshold 0:", (y_score > threshold))
threshold = 2000
print("Predicción con threshold 2000:", (y_score > threshold))

from sklearn.metrics import precision_recall_curve
y_scores = cross_val_predict(sgd, x_train, y_train_c, cv=3, method="decision_function")
precisions, recalls, umbrales = precision_recall_curve(y_train_c, y_scores)
plt.plot(umbrales, precisions[:-1], "b--", label="Precisión")
plt.plot(umbrales, recalls[:-1], "g-", label="Recall")
plt.show()
plt.plot(precisions[:-1], recalls[:-1], "g-", label="Precisión Vs Recall")
plt.xlabel("Recall")
plt.ylabel("Precisión")
plt.legend()
plt.show()

umbral_90 = umbrales[np.argmax(precisions >= 0.90)]
print("Umbral para 90% precisión:", umbral_90)
y_train_90 = (y_scores >= umbral_90)
print(confusion_matrix(y_train_c, y_train_90))
print("Precisión 90%:", precision_score(y_train_c, y_train_90), "Recall 90%:", recall_score(y_train_c, y_train_90))

# Curva ROC
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, umbrales_roc = roc_curve(y_train_c, y_scores)
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
plt.grid()
plt.show()
print("AUC SGD:", roc_auc_score(y_train_c, y_scores))

y_forest = cross_val_predict(rfc, x_train, y_train_c, cv=3, method="predict_proba")
y_scores_forest = y_forest[:, 1]
fpr_forest, tpr_forest, umbral_forest = roc_curve(y_train_c, y_scores_forest)
plt.plot(fpr, tpr, label="SGD ROC Curve")
plt.plot(fpr_forest, tpr_forest, label="RF ROC Curve")
plt.plot([0, 1], [0, 1], 'k--')
plt.legend()
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.grid()
plt.show()
print("AUC RF:", roc_auc_score(y_train_c, y_scores_forest))

# 7. Clasificadores Multiclase
from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train, y_train)
print(svm.predict([[25,0,1,0,167]]))
some_scores = svm.decision_function([[25,0,1,0,167]])
print("Decision function SVC:", some_scores)
print("Clase predicha:", np.argmax(some_scores))

from sklearn.multiclass import OneVsRestClassifier
svm_ovr = OneVsRestClassifier(SVC())
svm_ovr.fit(x_train, y_train)
print(svm_ovr.predict([[25,0,1,0,167]]))
some_scores_ovr = svm_ovr.decision_function([[25,0,1,0,167]])
print("Decision function OneVsRest:", some_scores_ovr)

# 8. Analizar Errores
y_train_pred_rf_multi = cross_val_predict(rfc, x_train, y_train, cv=3)
conf_mz = confusion_matrix(y_train, y_train_pred_rf_multi)
print("Matriz de confusión multiclase RF:\n", conf_mz)

# Clasificación binaria para Na_to_K_Bigger_Than_15
df['Na_to_K_Bigger_Than_15'] = [1 if i >= 15.015 else 0 for i in df.Na_to_K]
sgd = SGDClassifier(max_iter=1000, tol=1e-3)
sgd.fit(x_train, df.loc[x_train.index, 'Na_to_K_Bigger_Than_15'])
y_train_pred_nk = cross_val_predict(sgd, x_train, df.loc[x_train.index, 'Na_to_K_Bigger_Than_15'], cv=3)
conf_mz_nk = confusion_matrix(df.loc[x_train.index, 'Na_to_K_Bigger_Than_15'], y_train_pred_nk)
print("Matriz de confusión Na_to_K_Bigger_Than_15:\n", conf_mz_nk)

# Ejemplo de clasificación multilabel
y_0 = (y_train == 0)
y_5 = (y_train == 5)
y_multi = np.c_[y_0, y_5]
print("y_multi ejemplo:\n", y_multi)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_multi)
print(knn.predict([[45,0,1,0,89]]))

y_train_pred_knn = cross_val_predict(knn, x_train, y_multi, cv=3)
print("F1 macro:", f1_score(y_multi, y_train_pred_knn, average="macro"))
print("F1 weighted:", f1_score(y_multi, y_train_pred_knn, average="weighted"))
