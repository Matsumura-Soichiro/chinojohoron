import numpy as np
import pandas as pd
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem, Draw
import os


class gen_data:
    def __init__(self, path):
        self.path = path
        self.mols = self.get_mols()
        self.adjs = self.get_adjs()
        self.feature = self.get_features()
        self.inc = self.get_inc()
        self.atoms = self.get_atoms()

    def get_mols(self):
        path2 = self.path + "/molecule.csv"
        df = pd.read_csv(path2)
        self.df = df

        df_list = df["smiles"].values
        smiles = df_list.reshape(len(df_list))
        self.size_data = len(smiles)
        return [Chem.MolFromSmiles(sm) for sm in smiles]

    def createNodeFeatures(self, mol):
        """ノード特徴量を生成する

        :param mol: rdkit mol object
        :return: np.array
        """
        features = np.array(
            [
                [
                    a.GetIsAromatic(),
                    a.GetNoImplicit(),
                    a.IsInRing(),
                    a.GetAtomicNum(),
                    a.GetDegree(),
                    a.GetExplicitValence(),
                    a.GetImplicitValence(),
                    a.GetFormalCharge(),
                    a.GetTotalNumHs(),
                ]
                for a in mol.GetAtoms()
            ],
            dtype=np.int32,
        )

        return features

    def createAtomNumber(self, mol):
        return np.array([a.GetAtomicNum() for a in mol.GetAtoms()])

    def get_edge_index(self, x):
        y0 = []
        y1 = []
        for i in range(len(x)):
            for j in range(len(x)):
                if x[i][j] == 1:
                    y0.append(i)
                    y1.append(j)
        return np.array([y0, y1])

    def get_adjs(self):
        self.adjs = [
            pd.DataFrame(Chem.rdmolops.GetAdjacencyMatrix(mol)) for mol in self.mols
        ]
        return self.adjs

    def save_adjs(self):
        if self.adjs is not None:
            self.df['adjs'] = self.adjs

    def get_features(self):
        self.feature = [pd.DataFrame(self.createNodeFeatures(mol)) for mol in self.mols]
        return self.feature

    def save_features(self):
        if self.feature is not None:
            self.df['feature'] = self.feature

    def get_inc(self):
        self.inc = [pd.DataFrame(self.get_edge_index(adj)) for adj in self.adjs]
        return self.inc

    def save_inc(self):
        if self.inc is not None:
            self.df['inc'] = self.inc

    def get_atoms(self):
        self.atoms = [pd.Series(self.createAtomNumber(mol)) for mol in self.mols]
        return self.atoms

    def save_atoms(self):
        if self.atoms is not None:
            self.df['atoms'] = self.atoms

    def save_data(self):
        PICLKE_PATH = self.path+"/molecule.pickle'
        self.df.to_pickle(PICLKE_PATH)


class load_data:
    def __init__(self, path, data_size):
        self.path = path
        self.size_data = data_size

    def load_df(self):
        PICLKE_PATH = self.path + '/molecule.pickle'
        return pd.read_pickle(PICLKE_PATH)


def renumber_atom(x):
    x = np.where(x == 5, 0, x)
    x = np.where(x == 6, 1, x)
    x = np.where(x == 7, 2, x)
    x = np.where(x == 8, 3, x)
    x = np.where(x == 9, 4, x)
    x = np.where(x == 14, 5, x)
    x = np.where(x == 15, 6, x)
    x = np.where(x == 16, 7, x)
    x = np.where(x == 17, 8, x)
    x = np.where(x == 34, 9, x)
    x = np.where(x == 35, 10, x)
    x = np.where(x == 53, 11, x)
    return x


def reverse_atom(x):
    x = np.where(x == 0, 5, x)
    x = np.where(x == 1, 6, x)
    x = np.where(x == 2, 7, x)
    x = np.where(x == 3, 8, x)
    x = np.where(x == 4, 9, x)
    x = np.where(x == 5, 14, x)
    x = np.where(x == 6, 15, x)
    x = np.where(x == 7, 16, x)
    x = np.where(x == 8, 17, x)
    x = np.where(x == 9, 34, x)
    x = np.where(x == 10, 35, x)
    x = np.where(x == 11, 53, x)
    return x


def main():
    path = "../data"
    gen = gen_data(path)
    gen.save_adjs()
    gen.save_inc()
    gen.save_features()
    gen.save_atoms()
    gen.save_data()


if __name__ == "__main__":
    main()
