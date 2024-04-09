import numpy as np


def find_ids(all_data):
    ids = []
    for i in range(len(all_data)):
        ids.append(all_data[i][0])
        ids.append(all_data[i][1])

    unique_ids = np.unique(ids)
    np.random.shuffle(unique_ids)
    return unique_ids


def devide_ids(ids:list, k):
    new_ids = []
    for i in range(k):
        new_ids.append(ids[i::k])
    return new_ids

def devide_ids_new(ids:list, k):
    new_ids = []
    size_batch = int(len(ids) * 0.8)
    for i in range(k):
        new_ids.append(np.random.choice(ids, replace=False, size=size_batch))
    return new_ids


def create_dataset_task1(all_ids: list, all_data:list, k: int):
    dataset = dict()
    remain = set(range(0, len(all_data) - 1))
    size = int(len(all_data)/k)
    i = 0
    temp = []
    for id in all_ids:
        for j in remain:
            if all_data[j][0]==id  or  all_data[j][1]==id:
                temp.append(j)
        remain = remain.difference(temp)

        if len(temp) > size:       
            dataset[i] = set(temp)
            i+=1
            temp.clear()
            if (i==k-1):
                dataset[i]=remain
                return dataset
    return dataset




def prepare_smiles(drugs:list, smiles:list):
    drug_smiles = []
    for d in drugs:
        drug_smiles.append(smiles[d[0]])
    return np.array(drug_smiles)


    