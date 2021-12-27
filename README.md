# Projet Factorisation Matricielle
Il s'agit d'implémenter 2 modèles de prédiction pour une problématique de recommendation de contenus vidéo. Les prédictions portent sur l'[intérêt](https://colab.research.google.com/drive/14C8xy6_G6C_Pvv9qzbaRq1D9jFkDvlS3#scrollTo=0uyDSYsP8P05&line=1&uniqifier=1) exprimé par un client u sur un contenu i. La recommendation consiste à proposer au client une liste de contenus ordonnés selon l'intérêt estimé.

Les modèles à implémenter sont détaillés dans l'article [Koren08](https://dl.acm.org/doi/pdf/10.1145/1401890.1401944) à savoir:    

  1. **Baseline Estimates** (Section 2.1, équation (1));
  2. **SVD++** (Section 4, équation (15));

## Pré traitement des données d’entraînement et de test :
Le pré traitement des données s’est fait de la façon suivante, dans cet ordre:
<ul>
  <li>Calcul de l’intérêt en fonction du <em>dataframe ratings</em> et <em>favorites</em> (fonction interest-generator)
  <li>Séparation du train set et du test set en fonction des fichiers .npy contenant les indices à utiliser
</ul>

Pour la méthode <strong>Baseline Estimates</strong>, un traitement supplémentaire a été effectué:
<ul>
  <li>A partir du set d’entraînement, calcul de deux vecteurs rui-bu et rui-bi, chacun représentant respectivement la somme des lignes de rui (somme des intérêts liée aux utilisateurs) et la somme des colonnes de rui (somme des intérêts liée aux films)
  
  <li> Extraction des utilisateurs et films uniques
</ul>

Pour la méthode <strong> SVD++ </strong>, un traitement  supplémentaire a été effectué:
<ul>
  <li>A partir du set d’entraînement, calcul de <em>Nu</em>, une série Pandas contenant la liste de films dont l’utilisateur u
a exprimé un intérêt implicite.
  <li>A partir de du set d’entraînement, calcul de <em>Nu-count</em>, une série Pandas représentant le nombre de films dont
l’utilisateur u a exprimé un intérêt implicite.
  <li> Construction de dictionnaires mappant les ID des films/utilisateurs à leur indices respectifs dans les vecteurs
de paramètres leur correspondant.
  <li> Division du set d’entraînement en subdivisions afin de mieux contrôler le processus d’entraînement.
  <li> Transformation de la matrice <em>rui</em> en matrice sparse
</ul>

## Baseline Estimates

### 1. Description  

La méthode <strong> Baseline Estimates </strong> se base sur la prédiction de l’intérêt accordé par l’utilisateur u au film i en se basant sur trois paramètres :

<ul>
  <li> L’intérêt global mu
  <li> La déviation d’intérêt observée chez l’utilisateur u
  <li> La déviation d’intérêt observée chez le film i.
</ul>

Ainsi, on prend en considération les caractéristiques spécifiques des films (popularité) et utilisateurs (sévérité de notation) dans le système de recommandation au lieu de se baser uniquement sur la moyenne globale des intérêts. Ces déviations d’intérêts sont calculés via la minimisation de la moyenne des carrés des erreurs entre les valeurs d’intérêt prédits et les valeurs d’intérêt réels, plus un facteur de régularisation. La mise à jour des paramètres se fait par descente de gradients.


### 1. Résultats

Après entraînement sur 30 itérations, on constate une convergence de la fonction de coût, la dérivée de celle-ci atteignant 0.

Ci-dessous l'évolution de la moyenne de la dérivée de la fonction de cot pour les paramètres bu et bi.
![Baseline estimate!](/pics/be.PNG)

Un RMSE de 0.2 est relevé après application du modèle sur les données de test.

## SVD++

### 1. Description 

La méthode <strong>SVD++</strong> se base partiellement sur les mêmes paramètres que <strong>Baseline Estimates</strong>, et fait intervenir en plus de cela des facteurs latents, qui sont des vecteurs liés aux préférences des utilisateurs vis-à-vis des films dont ces derniers ont exprimé un intérêt implicite.

Ces facteurs latents, en plus des déviations d’intérêt, sont calculés via estimation, l’article <strong>"Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model"</strong> fournit les formules de mise à jour de ces paramètres via un processus itératif.

### 2. Résultats

Après entrainement sur 30 itérations, on constate une baisse notable de l’erreur relevée (RMSE).

Voici l'évolution de l’erreur quadratique (RMSE) pour chaque processus d’entraînement appliqué à chacune des subdivisions du set d’entraînement.

![SVD!](/pics/SVD.PNG)
