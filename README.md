# PFE-IA
#cellule 1 : importation des librairies nécessaire(numpy, pandas, csv, json...)
#cellule 2 : chemin du répertoir DATA
#cellule 3 : -Lecture du fichier .json
            -Création du dataframe à partir des données lues depuis le fichier json
            -Algo de récupération des timestamp à partir des images-ID (date et heure de prise de la mesure) :convertion en timestamp
            -Cas où les images-ID sont manquants : calcul d'un delta =0.87 l'intervalle de temps entre deux mesures
            -Dataframe ok avec l'ensemble des données entrantes pour le réseau
            
#cellule 4: On tronque les valeurs des timestamp qui s'avèrent trop longues.

#cellule 5 : Création du mask à partir du dataframe : 
def create_mask : 
'param : df : dataframe':
extrait la colonne timestamp qui nous intérresse pas et on garde le reste des colonnes. Condition de vérification, pour chaque mesure si la donnée est "not null/Nan" on encode 1 et 0 sinon.
return la matrice 1/0 de toutes les colonnes sauf "timestamp"

#cellule 6:
"all_timestamp" : on récupère la colonne timestamp avec toutes les valeurs qu'elle contient.
"processed_data" : création d'une structure à deux champs : un vecteur contenant les timestamp, et le mask créer à partir de la fonction "create_mask".

#cellule 7: préaparation des types de label à prédire sur les situations à risque qu'on peut avoir (risk_of_fall, risk_of_domestic_accident,full, other... )
affichage de la première ligne du dataframe : vérification de la cohérence des données du vecteur label_phase

#cellule 8 : -On désigne l'emplacement du répertoir de stockage des données
-création du repertoir de sauvegarde des données traitées.
Vérification si le répertoire de sauvegarde et bien crée ou non.

#Cellule 9 : sauvegarde des 3 vecteurs d'entrées au réseau : timestamp, all_input_data/mask et le vecteur des labels à prédire: data.npz

#Cellule 10: Création du processed_fold : 
Initialisation des vecteurs processed_fold et processed_stats.

processed_fold : correspond à la structure contenant les données d'apprentissage, où n_samples est le nombre d'échantillons et n_features est le nombre d'entités et le nombre d'échantillonnage.

processed_stats : correspond au données statistiques, soit la moyenne et l'écartype des échantillons.


#Cellule 11 : 
"ts = TimeSeriesSplit(n_splits=5)" :
Time Series cross-validator

Provides train/test indices to split time series data samples that are observed at fixed time intervals, in train/test sets. In each split, test indices must be higher than before, and thus shuffling in cross validator is inappropriate.

This cross-validation object is a variation of KFold. In the kth split, it returns first k folds as train set and the (k+1)th fold as test set.

Note that unlike standard cross-validation methods, successive training sets are supersets of those that come before them.
https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation

#Cellule 12: sauvegarde : fold.npz, contenant la validation croisée: évaluation des performances de l'estimateur.


#celluel 13: On load le fichier data.npz et fold.npz.
on vérifie pour chaque clé la donnée enregistrée : en "all_variables_data"




