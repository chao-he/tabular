import rdkit.Chem as Chem
from rdkit.Chem import Descriptors, Lipinski


def calcMolAttr(smiles):
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    return [round(Descriptors.MolWt(m)),
            len(m.GetAtoms()),
            len([a for a in m.GetAtoms() if a.GetAtomicNum() == 1]),
            Descriptors.HeavyAtomCount(m),
            Lipinski.RingCount(m),
            Lipinski.NumRotatableBonds(m),
            Lipinski.NumAromaticRings(m),
            Descriptors.NumHAcceptors(m),
            Descriptors.NumHDonors(m),
            Descriptors.TPSA(m),
            Descriptors.MolLogP(m)
            ]

if __name__ == "__main__":
    r = calcMolAttr('c1ccccc1C(=O)O')
    print(r)
