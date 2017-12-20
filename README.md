#README#


----------

 - Charger les données avec prétraitement des Na et des variables categorielles

```python
from load_data import *

y_train, X_train, X_test, test_ids = ultimeload("../pathTrain", "../pathTest")
```

- Entrainer son modèle sur *X_train* avec pour cible *y_train*. Une fois les paramètres du modèle fixés l'appliquer sur *X_test* pour obtenir *y_test*.

- Enregistrer sa submission en utilisant  *createSubmission*, toujours définie dans le fichier **load_data**:

``` python
createSubmission("SubmissionName.csv", y_test, test_ids)
```

- Il ne reste plus qu'à la soumettre sur Kaggle et voir son nouveau score !


 - conservez vos submissions, ca servira à faire une grande moyenne à la fin !
