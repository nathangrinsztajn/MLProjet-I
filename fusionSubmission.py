import pandas as pd

def fusionneTab(tableau):
    result = pd.DataFrame()
    result['id'] = tableau[0]['id']
    result['target'] = tableau[0]['target']
    for i in tableau[1:]:
        result['target'] = result['target']+i['target']
    result['target'] = result['target']/len(tableau)
    return result

def fusionPondérées(targets, poids):
    result = pd.DataFrame()
    result['id'] = targets[0]['id']
    result['target'] = targets[0]['target']*poids[0]
    for i in range(1, len(targets)):
        result['target'] = result['target']+targets[i]['target']*poids[i]
    result['target'] = result['target']/sum(poids)
    return result

