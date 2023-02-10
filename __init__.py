import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pandas as pd
from pandas import DataFrame
from bs4 import BeautifulSoup, SoupStrainer
import urllib.request
import numpy as np
import os
import itertools

pd.options.mode.chained_assignment = None
from rdkit.Chem.Descriptors3D import Asphericity, Eccentricity, InertialShapeFactor, NPR1, NPR2, PMI1, PMI2, PMI3, \
    RadiusOfGyration, SpherocityIndex
from rdkit.Chem import AllChem, rdDistGeom
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from psikit import Psikit
from bs4 import BeautifulSoup, SoupStrainer
import urllib.request

pd.options.mode.chained_assignment = None
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, \
    GradientBoostingClassifier, ExtraTreesClassifier
import glob
from sklearn.ensemble import VotingRegressor
from pandas.plotting import scatter_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
import itertools
from pandas import DataFrame, Series
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
# RDkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# Pytorch and Pytorch Geometric
import torch
import torch_geometric
from torch.nn import Linear
from torch_geometric.data import Data, DataLoader
import torch_geometric.nn as nn
from torch_geometric.nn import GCNConv

import pickle
import traceback
import sys
import matplotlib.pyplot as plt

class ILP(object):
    def __init__(self):
        # Set room temperature to 25°C
        T_norm = 25
        self.T_norm = T_norm
        # Set the self.feature_list for machine learning

        self.feature_list = ['Asphericity_anion', 'Eccentricity_anion',
                        'InertialShapeFactor_anion', 'NPR1_anion', 'NPR2_anion', 'PMI1_anion',
                        'PMI2_anion', 'PMI3_anion', 'RadiusOfGyration_anion',
                        'SpherocityIndex_anion', 'ExactMolWt_anion', 'FpDensityMorgan1_anion',
                        'FpDensityMorgan2_anion', 'HeavyAtomMolWt_anion',
                        'MaxAbsPartialCharge_anion', 'MaxPartialCharge_anion',
                        'MinPartialCharge_anion', 'NumRadicalElectrons_anion',
                        'NumValenceElectrons_anion', 'volume_anion',
                        'Asphericity_cation', 'Eccentricity_cation',
                        'InertialShapeFactor_cation', 'NPR1_cation', 'NPR2_cation',
                        'PMI1_cation', 'PMI2_cation', 'PMI3_cation', 'RadiusOfGyration_cation',
                        'SpherocityIndex_cation', 'ExactMolWt_cation',
                        'FpDensityMorgan1_cation', 'FpDensityMorgan2_cation',
                        'HeavyAtomMolWt_cation', 'MaxAbsPartialCharge_cation',
                        'MaxPartialCharge_cation', 'MinPartialCharge_cation',
                        'NumRadicalElectrons_cation', 'NumValenceElectrons_cation',
                        'volume_cation', 'Asphericity_pair',
                        'Eccentricity_pair', 'InertialShapeFactor_pair', 'NPR1_pair',
                        'NPR2_pair', 'PMI1_pair', 'PMI2_pair', 'PMI3_pair',
                        'RadiusOfGyration_pair', 'SpherocityIndex_pair', 'ExactMolWt_pair',
                        'FpDensityMorgan1_pair', 'FpDensityMorgan2_pair', 'HeavyAtomMolWt_pair',
                        'MaxAbsPartialCharge_pair', 'MaxPartialCharge_pair',
                        'MinPartialCharge_pair', 'NumRadicalElectrons_pair',
                        'NumValenceElectrons_pair', 'volume_pair', 'energy_anion', 'HOMO_anion', 'LUMO_anion',
                        'dipole_x_anion', 'dipole_y_anion', 'dipole_z_anion',
                        'energy_cation', 'HOMO_cation', 'LUMO_cation', 'dipole_x_cation',
                        'dipole_y_cation', 'dipole_z_cation'
                        ]

        self.threeD_colum = ["Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2", "PMI1", "PMI2",
                             "PMI3",
                             "RadiusOfGyration", "SpherocityIndex"]
        self.mol_colum = ["ExactMolWt", "FpDensityMorgan1", "FpDensityMorgan2", "HeavyAtomMolWt", "MaxAbsPartialCharge",
                          "MaxPartialCharge", "MinPartialCharge", "NumRadicalElectrons", "NumValenceElectrons",
                          "volume"]
        self.abs_path = os.path.dirname(__file__)

    # We use dataCrawler() to crawl and clean the data about positive and negative ions and ionic liquids.
    # These data will be stored in the dataset.
    def dataCrawler(self, parser="html.parser", url="https://iolitec.de/en/products/list"):
        print("Crawl the data from the website!")
        resp = urllib.request.urlopen(url)
        soup = BeautifulSoup(resp, parser, from_encoding=resp.info().get_param('charset'))
        print("Crawl the data from the website!")
        # Create list to store data
        product_link = []

        # Crawl the information of "/ionic_liquids/catalogue/" and append it to the product_link list
        for link in soup.find_all('a', href=True):
            if "/ionic_liquids/catalogue/" in link['href']:
                product_link.append(link['href'])

        # Create a dictionary with the name of the ionic liquid as the key
        results_dict = {}
        for link in product_link:
            ans_resp = urllib.request.urlopen("https://iolitec.de" + link)
            soup = BeautifulSoup(ans_resp, parser, from_encoding=ans_resp.info().get_param('charset'))
            label_cont = soup.find('h1')
            label = label_cont.text.split(", ")[0]
            results_dict[label] = []
            for res in soup.findAll(class_='col-2td'):
                for item in res.findAll('div'):
                    results = item.text
                    if results != 'Produkt Nr.:' and results != 'Product Nr.:':
                        results_dict[label].append(results)
        feature = []  # Store ionic liquid attributes
        value = []  # Store ionic liquid attribute values
        label = []  # Store ionic liquid name

        # Handle abnormal data
        for keys, values in results_dict.items():
            for j in range(0, len(values)):
                if keys != "	Guanidinium iodide":  # Exclude some unusual cations
                    if j % 2 == 1:
                        value.append(values[j])
                        label.append(keys)
                    elif j % 2 == 0:
                        feature.append(values[j][0:-1])
                else:
                    pass
        dataframe = DataFrame({"label": label, "feature": feature, "value": value})

        dataframe = dataframe.drop_duplicates()  # Remove all duplicate values
        pivot_table = dataframe.pivot(values='value', index='label', columns='feature')
        final_df = pd.DataFrame(pivot_table.to_records())

        # Fill the final_df  with "" for NaN
        final_df["conductivity"] = final_df["Conductivity"].fillna('') + final_df["Leitfähigkeit"].fillna('')
        final_df["density"] = final_df["Density"].fillna('') + final_df["Dichte"].fillna('')
        final_df["melting_point"] = final_df["Melting point"].fillna('') + final_df["Schmelzpunkt"].fillna('')
        final_df["meolecular_weight"] = final_df["Molecular weight"].fillna('') + final_df["Molekulargewicht"].fillna(
            '')
        final_df["viscosity"] = final_df["Viscosity"].fillna('') + final_df["Viskosität"].fillna('')
        final_df["formula"] = final_df["Sum formula"].fillna('') + final_df["Summenformel"].fillna('')

        # Create DataFrame daraset1
        dataset1 = final_df[
            ['label', 'CAS Nr.', 'conductivity', 'ECW', 'density', 'melting_point', 'meolecular_weight', 'viscosity',
             'formula']]

        # Delete spaces, line breaks, etc.
        dataset1['label'] = dataset1['label'].str[1:]
        dataset1['label'] = dataset1['label'].str.lstrip()
        dataset1['viscosity'] = dataset1['viscosity'].str.lstrip()
        dataset1['conductivity'] = dataset1['conductivity']
        dataset1['density'] = dataset1['density'].str.lstrip()

        dataset1["T_conductivity"] = dataset1["conductivity"].str.split("(", expand=True)[1]
        dataset1["T_conductivity"] = dataset1["T_conductivity"].str.split(" ", expand=True)[0]
        dataset1["conductivity"] = dataset1["conductivity"].str.split(" ", expand=True)[0]

        dataset1["T_density"] = dataset1["density"].str.split("(", expand=True)[1]
        dataset1["T_density"] = dataset1["T_density"].str.split(" ", expand=True)[0]
        dataset1["density"] = dataset1["density"].str.split(" ", expand=True)[0]

        dataset1["T_viscosity"] = dataset1["viscosity"].str.split("(", expand=True)[1]
        dataset1["T_viscosity"] = dataset1["T_viscosity"].str.split(" ", expand=True)[0]
        dataset1["viscosity"] = dataset1["viscosity"].str.split(" ", expand=True)[0]

        dataset1["ECW"] = dataset1["ECW"].str.split(" ", expand=True)[0]
        dataset1['formula'] = dataset1['formula'].str.split('\n', expand=True)[0]

        dataset = dataset1

        # Replace ""  in the dataset with NaN
        dataset.replace('', np.nan, inplace=True)

        # Remove spaces from the left side of the data
        dataset['label'] = dataset['label'].str.lstrip()
        dataset['viscosity'] = dataset['viscosity'].str.lstrip()
        dataset['conductivity'] = dataset['conductivity'].str.lstrip()
        dataset['density'] = dataset['density'].str.lstrip()

        dataset["density"] = dataset["density"].str.split(" ", expand=True)
        dataset["viscosity"] = dataset["viscosity"].str.split(" ", expand=True)
        dataset["melting_point"] = dataset["melting_point"].str.split(" ", expand=True)[0]

        dataset["label"] = dataset["label"].str.split(",>", expand=True, n=1)[0]
        dataset["cation"] = dataset["label"].str.split(" ", expand=True, n=1)[0]
        dataset["anion"] = dataset["label"].str.split(" ", expand=True, n=1)[1]
        dataset["anion"] = dataset["anion"].str.lstrip()
        dataset["anion"] = dataset["anion"].str.split(",>", expand=True, n=1)[0]

        dataset['formula'] = dataset['formula'].str.split('\n', expand=True)

        dataset = dataset.drop(['CAS Nr.'], axis=1)

        dataset[['meolecular_weight', 'viscosity', 'conductivity', 'density']].replace('', np.nan, inplace=True)

        dataset = dataset.drop(['meolecular_weight'], axis=1)

        dataset['T_density'] = dataset['T_density'].str.extract('(\d+)', expand=False)

        # Convert data types to float
        dataset[['viscosity', 'conductivity', 'density', 'T_conductivity', 'T_density', 'T_viscosity']] = dataset[
            ['viscosity', 'conductivity', 'density', 'T_conductivity', 'T_density', 'T_viscosity']].astype(float)

        #Keep the FSI anion by manually adding the melting_point the conductivity
        dataset.loc[dataset[dataset['anion'] == 'bis(fluorosulfonyl)imide']]['melting_point'] = '>RT'

        # Delete NaN value
        dataset = dataset.dropna(subset=['melting_point', 'anion'], axis=0)

        # Correct misspellings
        dataset = dataset.loc[dataset['cation'] != '1-Hexyl-1,4-diaza[2.2.2]bicyclooctanium']
        dataset = dataset.loc[dataset['cation'] != 'Bis(1-butyl-3-methylimidazolium)']
        dataset = dataset.loc[dataset['cation'] != 'Bis(1-ethyl-3-methylimidazolium)']
        dataset = dataset.loc[dataset['cation'] != 'Tetraoctylphosphonium']
        dataset = dataset.loc[dataset['cation'] != 'Tetraoctylphosphonium']

        dataset['anion'] = dataset['anion'].str.replace("methyl  sulfate", "methyl sulfate")
        dataset['anion'] = dataset['anion'].str.replace("methylsulfate", "methyl sulfate")
        dataset['anion'] = dataset['anion'].str.replace("chloride >98%\n", "chloride")
        dataset['anion'] = dataset['anion'].str.replace("tricyanomethanide >98%\n", "tricyanomethanide")
        #dataset['anion'] = dataset['anion'].str.replace("bis(trifluoromethylsulfonyl)imid", "bis(trifluoromethylsulfonyl)imide")
        dataset = dataset.loc[dataset['anion'] != 'Tetraoctylphosphonium']
        #dataset = dataset.loc[dataset['anion'] != 'bis(trifluoromethylsulfonyl)imid']
        dataset = dataset.loc[dataset['anion'] != 'tetrachloroferrate(III)']
        dataset = dataset.loc[dataset['anion'] != 'iodide']
        dataset["anion"] = dataset["anion"].str.strip('\n')
        dataset['anion'] = dataset['anion'].str.replace("bis(trifluoromethylsulfonyl)imid",
                                                        "bis(trifluoromethylsulfonyl)imide")
        dataset.at[dataset[
                       'label'] == "Trimethylsulfonium bis(trifluoromethylsulfonyl)imid", 'anion'] = "bis(trifluoromethylsulfonyl)imide"
        dataset.to_csv(os.path.join(self.abs_path, "output", "dataset_iolitech.csv"), index=None)

        return dataset

    # Before performing the calculation, we need to further process the data. We do this step with dataProcessing().
    def dataProcessing(self):
        # Verify the existence of the predecessor file, if it exists, then read it directly, if not, then get it through the function
        if os.path.exists(os.path.join(self.abs_path, "output", "dataset_iolitech.csv")) == True:
            dataset = pd.read_csv(os.path.join(self.abs_path, "output", "dataset_iolitech.csv"), index_col=None)
        elif os.path.exists(os.path.join(self.abs_path, "output", "dataset_iolitech.csv")) == False:
            dataset = self.dataCrawler()

        # When the test temperature is NaN, set the test temperature to 25°C
        dataset[['T_conductivity', 'T_density', 'T_viscosity']] = dataset[
            ['T_conductivity', 'T_density', 'T_viscosity']].fillna(self.T_norm)

        # Get the 'conductivity', 'viscosity', and 'density' at room temperature（25°C）
        # dataset[['conductivity', 'viscosity', 'density']] = dataset[['conductivity', 'viscosity', 'density']].fillna(0)

        # If 'conductivity_norm', 'viscosity_norm', 'density_norm' is 0, then assign to None
        dataset['conductivity_norm'] = dataset['conductivity'] / dataset['T_conductivity'] * self.T_norm
        dataset['viscosity_norm'] = dataset['viscosity'] / dataset['T_viscosity'] * self.T_norm
        dataset['density_norm'] = dataset['density'] / dataset['T_density'] * self.T_norm

        # dataset[['conductivity_norm', 'viscosity_norm', 'density_norm']] = \
        # dataset[['conductivity_norm', 'viscosity_norm', 'density_norm']].replace(0, None)

        new_unique_cation = dataset["cation"].unique()
        new_unique_anion = dataset["anion"].unique()

        dataset = dataset.reset_index(drop=True)
        cation_kind = ['imidazolium', 'pyridinium', 'piperidinium', 'pyrrolidinium',
                       'ammonium', 'sulfonium', 'phosphonium', 'Choline']

        anion_kind = ['imid', 'tetrafluoroborate',
                      'triflate',
                      'sulfate', 'dicyanamide', 'tricyanomethanide', 'thiocyanate',
                      'bromide', 'chloride', 'acetate',
                      'sulfonate', 'tosylate', 'formate',
                      'nitrate', 'phosph', 'decanoate', 'carbonate', 'cobaltate', 'hydroxide']
        cation_type = []
        anion_type = []

        # Classify all ions in the dataset
        for i in range(0, len(dataset)):
            if len(cation_type) != len(anion_type):
                print(i)
            for kind in cation_kind:
                dummy = len(cation_type)
                if kind in dataset['cation'][i]:
                    cation_type.append(kind)
                    break

            for kind in anion_kind:
                if kind in dataset['anion'][i]:
                    anion_type.append(kind)
                    break

        # Create new columns 'cation_type' and 'anion_type' in the dataset
        dataset['cation_type'] = cation_type
        dataset['anion_type'] = anion_type


        dataset.to_csv(os.path.join(self.abs_path, "output", "dataset_iolitech_type.csv"), index=False)

        return dataset

    # Rdkit descriptors
    def cal3Ddescriptor(self, sml):
        try:
            # Exception for PF6- anion, the RDKit is currently not able to reliably generate conformer for octahedral
            # species. Thus, we will use PF5- to make an approximate. We are not willing to remove thus anion, since
            # the ionic liquids with PF6- anion count for a large population in the dataset pool.
            # THus, we would like to keep this anion in order to improve the accuracy of the model and diversity of the
            # sample set. Hopefully, This issue could be resolved in the future Rdkit version.
            if "F[P-](F)(F)(F)(F)F" in sml:
                sml = sml.replace("F[P-](F)(F)(F)(F)F", "F[P-](F)(F)(F)F")
                print(sml, "occured")
            print(sml)
            # Convert SMILES structure to molecular formula
            m = Chem.MolFromSmiles(sml)
            # Transform the 2D molecular map into 3D molecular coordinates
            AllChem.EmbedMolecule(m, useRandomCoords=True, maxAttempts=5000)
            # Use UFFO for conformational optimization of molecule
            AllChem.UFFOptimizeMolecule(m)
            # Add Hs to molecule
            m2 = Chem.AddHs(m)
            # Generate 3d conformation of the molecule and optimize it using ETKDG algorithm
            AllChem.EmbedMolecule(m2, AllChem.ETKDG())
            # Ccalculate 3D molecular descriptors
            Descriptors_3d = []
            Descriptors_3d.append(Asphericity(m2))
            Descriptors_3d.append(Eccentricity(m2))
            Descriptors_3d.append(InertialShapeFactor(m2))
            Descriptors_3d.append(NPR1(m2))
            Descriptors_3d.append(NPR2(m2))
            Descriptors_3d.append(PMI1(m2))
            Descriptors_3d.append(PMI2(m2))
            Descriptors_3d.append(PMI3(m2))
            Descriptors_3d.append(RadiusOfGyration(m2))
            Descriptors_3d.append(SpherocityIndex(m2))
        except:
            print("Failure to embed molecule!")
            Descriptors_3d = [] * 10
        return Descriptors_3d

    # Rdkit 3D descriptors
    def calMoldescriptor(self, sml):  # This method can be used individually
        try:
            # Exception for PF6- anion, the RDKit is currently not able to reliably generate conformer for octahedral
            # species. Thus, we will use PF5- to make an approximate. We are not willing to remove thus anion, since
            # the ionic liquids with PF6- anion count for a large population in the dataset pool.
            # THus, we would like to keep this anion in order to improve the accuracy of the model and diversity of the
            # sample set. Hopefully, This issue could be resolved in the future Rdkit version.
            if "F[P-](F)(F)(F)(F)F" in sml:
                sml = sml.replace("F[P-](F)(F)(F)(F)F", "F[P-](F)(F)(F)F")
                print(sml, "occured")
            print("3D", sml)
            m = Chem.MolFromSmiles(sml)
            AllChem.EmbedMolecule(m, useRandomCoords=True, maxAttempts=5000)
            AllChem.UFFOptimizeMolecule(m)
            m2 = Chem.AddHs(m)
            AllChem.EmbedMolecule(m2, AllChem.ETKDG())
            # Calculate molecular descriptors
            Descriptors_mol = MolecularDescriptorCalculator(
                ["ExactMolWt", "FpDensityMorgan1", "FpDensityMorgan2", "HeavyAtomMolWt", "MaxAbsPartialCharge",
                 "MaxPartialCharge", "MinPartialCharge", "NumRadicalElectrons",
                 "NumValenceElectrons"]).CalcDescriptors(
                m2)
            Descriptors_mol_list = list(Descriptors_mol)
            Descriptors_mol_list.append(AllChem.ComputeMolVolume(m2))
        except:
            Descriptors_mol_list = [] * 10
        return Descriptors_mol_list

    # Psi4 calculate the molecular energy, HOMO, LUMO, coordinates and dipole moment
    def calMolEnergy(self, sml, basis_set="b3lyp/6-311pg**"):
        print(basis_set)
        pk = Psikit()  # Instantiate Psikit()
        pk.read_from_smiles(sml)
        energy = pk.optimize(basis_set)
        # pk.frequency("b3lyp/6-31g*")
        x, y, z, total = pk.dipolemoment  # Calculate the coordinates and dipole moments
        return [energy, pk.HOMO, pk.LUMO, x, y, z,
                total]  # Returns molecular energy, HOMO, LUMO, atomic coordinates and dipole moment

    # Now we need to get the descriptors of the molecules using despGenerator() mainly based on RDKit
    def despGenerator(self):
        # Verify the existence of the predecessor file, if it exists, then read it directly, if not, then get it through the function
        if os.path.exists(os.path.join(self.abs_path, "output", "dataset_iolitech_type.csv")) == True:
            dataset = pd.read_csv(os.path.join(self.abs_path, "output", "dataset_iolitech_type.csv"), index_col=None)
        elif os.path.exists(os.path.join(self.abs_path, "output", "dataset_iolitech.csv")) == True and os.path.exists(
                os.path.join(self.abs_path, "output", "dataset_iolitech_type.csv")) == False:
            dataset = pd.read_csv(os.path.join(self.abs_path, "output", "dataset_iolitech.csv"), index_col=None)
            dataset = self.data_processing()
        elif os.path.exists(os.path.join(self.abs_path, "output", "dataset_iolitech.csv")) == False and os.path.exists(
                os.path.join(self.abs_path, "output", "dataset_iolitech_type.csv")) == False:
            dataset = self.Crawler()
            dataset = self.data_processing()

        # Define function "calDescriptor", used to calculate the positive and negative ions of threeDdescriptor and Moldescriptor data
        def calDescriptor(cation_smiles, anion_smiles):

            threeDdescriptor = []
            Moldescriptor = []

            for i in range(0, len(cation_smiles)):
                sml = cation_smiles["cation_sml"][i]
                Moldescriptor.append(self.calMoldescriptor(sml))
                threeDdescriptor.append(self.cal3Ddescriptor(sml))

            cation_smiles[[col + "_cation" for col in self.threeD_colum]] = DataFrame(threeDdescriptor,
                                                                                      columns=[col + "_cation" for col
                                                                                               in
                                                                                               self.threeD_colum])
            cation_smiles[[col + "_cation" for col in self.mol_colum]] = DataFrame(Moldescriptor,
                                                                                   columns=[col + "_cation" for col in
                                                                                            self.mol_colum])

            threeDdescriptor = []
            Moldescriptor = []
            for i in range(0, len(anion_smiles)):
                sml = anion_smiles["anion_sml"][i]
                Moldescriptor.append(self.calMoldescriptor(sml))
                threeDdescriptor.append(self.cal3Ddescriptor(sml))

            # Store the calculation results in "anion_smiles"
            anion_smiles[[col + "_anion" for col in self.threeD_colum]] = DataFrame(threeDdescriptor,
                                                                                    columns=[col + "_anion" for col in
                                                                                             self.threeD_colum])
            anion_smiles[[col + "_anion" for col in self.mol_colum]] = DataFrame(Moldescriptor,
                                                                                 columns=[col + "_anion" for col in
                                                                                          self.mol_colum])

        # Define the function calPairDescriptor for calculating the threeDdescriptor and Moldescriptor data for ion pairs
        def calPairDescriptor(input_dataset):

            threeDdescriptor = []
            Moldescriptor = []
            for i in range(0, len(input_dataset)):
                sml = input_dataset["pair_sml"][i]
                print(i, sml)
                threeDdescriptor.append(self.cal3Ddescriptor(sml))
                Moldescriptor.append(self.calMoldescriptor(sml))

            input_dataset[[col + "_pair" for col in self.threeD_colum]] = DataFrame(threeDdescriptor,
                                                                                    columns=[col + "_pair" for col in
                                                                                             self.threeD_colum])
            input_dataset[[col + "_pair" for col in self.mol_colum]] = DataFrame(Moldescriptor,
                                                                                 columns=[col + "_pair" for col in
                                                                                          self.mol_colum])

        # Read the smiles structure of cations and anions
        cation_smiles = pd.read_excel(os.path.join(self.abs_path, "output", "IL_smiles_iolitech.xlsx"),
                                      sheet_name='cation')
        anion_smiles = pd.read_excel(os.path.join(self.abs_path, "output", "IL_smiles_iolitech.xlsx"),
                                     sheet_name='anion')
        cation_smiles.columns = ['cation', 'cation_sml']
        anion_smiles.columns = ['anion', 'anion_sml']
        current_cation = list(cation_smiles['cation'])
        current_anion = list(anion_smiles['anion'])
        print(len(current_cation))
        print(len(current_anion))

        # convert "cation", "anion" column to lower case
        dataset["cation"] = dataset["cation"].str.lower()
        dataset["anion"] = dataset["anion"].str.lower()
        new_unique_cation = dataset["cation"].unique()
        new_unique_anion = dataset["anion"].unique()
        print(len(new_unique_cation))
        print(len(new_unique_anion))
        # Check ions for mismatches and missing
        for i in range(0, len(current_cation)):
            if current_cation[i] not in new_unique_cation:
                print("Extra", current_cation[i])
        for i in range(0, len(new_unique_cation)):
            if new_unique_cation[i] not in current_cation:
                print("Missing", new_unique_cation[i])
        for i in range(0, len(current_anion)):
            if current_anion[i] not in new_unique_anion:
                print("Extra", current_anion[i])
        for i in range(0, len(new_unique_anion)):
            if new_unique_anion[i] not in current_anion:
                print("Missing", new_unique_anion[i])

        # Permutate the cation and anion list
        # Create dataframe object of "input_dataset" to hold positive and negative ions, ion pairs and SMILES structure
        input_dataset = DataFrame({"pair": [r[0] + ' ' + r[1] for r in itertools.product(list(cation_smiles['cation']),
                                                                                         list(anion_smiles['anion']))],
                                   'cation': [r[0] for r in itertools.product(list(cation_smiles['cation']),
                                                                              list(anion_smiles['anion']))],
                                   'anion': [r[1] for r in itertools.product(list(cation_smiles['cation']),
                                                                             list(anion_smiles['anion']))],
                                   "pair_sml": [r[0] + '.' + r[1] for r in
                                                itertools.product(list(cation_smiles['cation_sml']),
                                                                  list(anion_smiles['anion_sml']))],
                                   'cation_sml': [r[0] for r in itertools.product(list(cation_smiles['cation_sml']),
                                                                                  list(anion_smiles['anion_sml']))],
                                   'anion_sml': [r[1] for r in itertools.product(list(cation_smiles['cation_sml']),
                                                                                 list(anion_smiles['anion_sml']))]})

        calDescriptor(cation_smiles, anion_smiles)

        # Merge "anion_smiles", "cation_smiles" and "input_dataset" all together

        input_dataset = pd.merge(cation_smiles, input_dataset, on=['cation', 'cation_sml'])
        print(len(input_dataset))
        input_dataset = pd.merge(anion_smiles, input_dataset, on=['anion', 'anion_sml'])

        calPairDescriptor(input_dataset)

        # Export input_dataset as "dataset_iolitech_rdkit.csv"
        input_dataset.to_csv(os.path.join(self.abs_path, "output", "dataset_iolitech_rdkit.csv"), index=False)

        return input_dataset

    # We use psiCal to perform quantum chemistry calculation of the cations and anions.
    def psiCal(self, basis_set):
        # Verify the existence of the predecessor file, if it exists, then read it directly, if not, then get it through the function
        if os.path.exists(os.path.join(self.abs_path, "output", "dataset_iolitech_rdkit.csv")) == True:
            dataset_iolitech_rdkit = pd.read_csv(os.path.join(self.abs_path, "output", "dataset_iolitech_rdkit.csv"),
                                                 index_col=None)
            print("OK")
        elif os.path.exists(
                os.path.join(self.abs_path, "output", "dataset_iolitech_type.csv")) == True and os.path.exists(
                os.path.join(self.abs_path, "output", "dataset_iolitech_rdkit.csv")) == False:
            dataset_iolitech_rdkit = pd.read_csv(os.path.join(self.abs_path, "output", "dataset_iolitech_type.csv"),
                                                 index_col=None)
            dataset_iolitech_rdkit = self.Descriptor()
        elif os.path.exists(os.path.join(self.abs_path, "output", "dataset_iolitech.csv")) == True and os.path.exists(
                os.path.join(self.abs_path, "output", "dataset_iolitech_type.csv")) == False and os.path.exists(
            os.path.join(self.abs_path, "output", "dataset_iolitech_rdkit.csv")) == False:
            dataset_iolitech_rdkit = pd.read_csv(os.path.join(self.abs_path, "output", "dataset_iolitech.csv"),
                                                 index_col=None)
            dataset_iolitech_rdkit = self.data_processing()
            dataset_iolitech_rdkit = self.escriptor()
        elif os.path.exists(os.path.join(self.abs_path, "output", "dataset_iolitech.csv")) == False and os.path.exists(
                os.path.join(self.abs_path, "output", "dataset_iolitech_type.csv")) == False and os.path.exists(
            os.path.join(self.abs_path, "output", "dataset_iolitech_rdkit.csv")) == False:
            dataset_iolitech_rdkit = self.dataCrawler()
            dataset_iolitech_rdkit = self.data_processing()
            dataset_iolitech_rdkit = self.Descriptor()

        #cation_smiles = pd.read_excel(os.path.join(self.abs_path,"output","IL_smiles_iolitech.xlsx"), sheet_name='cation')

        #anion_smiles = pd.read_excel(os.path.join(self.abs_path, "output", "IL_smiles_iolitech.xlsx"), sheet_name='anion')
        cation_smiles = pd.read_csv(os.path.join(self.abs_path, "output", "cation_smiles.csv"), index_col=None)
        anion_smiles = pd.read_csv(os.path.join(self.abs_path, "output", "anion_smiles.csv"), index_col=None)
        """
        cation_results = []

        # Use function calMolEnergy(sml) for cation

        for i in range(0, len(cation_smiles)):
            sml = cation_smiles['smile_form'][i]  # Obtain the SMILES structure of positive ion
            print(i, sml)
            try:
                cation_results.append(self.calMolEnergy(sml, basis_set))
            except:
                cation_results.append([0] * 7)
                traceback.print_exception(*sys.exc_info())
                print("Exception occured!")

        cation_smiles[["energy", "HOMO", "LUMO", "dipole_x", "dipole_y", "dipole_z", "dipole_total"]] = DataFrame(
            cation_results, columns=["energy", "HOMO", "LUMO", "dipole_x", "dipole_y", "dipole_z", "dipole_total"])

        cation_smiles.to_csv(os.path.join(self.abs_path, "output", "cation_smiles.csv"), index=False)

        anion_results = []
        # Use function calMolEnergy(sml) for anion
        for i in range(0, len(anion_smiles)):
            sml = anion_smiles['smiles_form'][i]
            print(i, sml)
            try:
                anion_results.append(calMolEnergy(sml, "scf/6-311pg**"))
                print(anion_results)
                anion_smiles[
                    ["energy", "HOMO", "LUMO", "dipole_x", "dipole_y", "dipole_z", "dipole_total"]] = DataFrame(
                    anion_results,
                    columns=["energy", "HOMO", "LUMO", "dipole_x", "dipole_y", "dipole_z", "dipole_total"])
                anion_smiles.to_csv("anion_smiles.csv", index=False)
                print(anion_smiles)
            except:
                anion_results.append([0, 0, 0, 0, 0, 0, 0])
                traceback.print_exception(*sys.exc_info())
                print("Exception occured!")
        """
        # Change the column index of anion_smiles, cation_smiles
        anion_smiles.columns = [item + '_anion' for item in
                                ['name', 'smiles_form', 'energy', 'HOMO', 'LUMO', "dipole_x", "dipole_y", "dipole_z",
                                 "dipole_total"]]
        cation_smiles.columns = [item + '_cation' for item in
                                 ['name', 'smiles_form', 'energy', 'HOMO', 'LUMO', "dipole_x", "dipole_y", "dipole_z",
                                  "dipole_total"]]

        # Check for missing data
        for anion in list(anion_smiles["name_anion"].unique()):
            if anion not in list(dataset_iolitech_rdkit["anion"].unique()):
                print("MIssing", anion)
        for anion in list(dataset_iolitech_rdkit["anion"].unique()):
            if anion not in list(anion_smiles["name_anion"].unique()):
                print("Extra", anion)

        # Merge datasets to get "dataset_iolitech_rdkit_psi4_final"
        dataset_iolitech_rdkit_psi4 = pd.merge(dataset_iolitech_rdkit, anion_smiles, left_on=['anion'],
                                               right_on=['name_anion'])
        dataset_iolitech_rdkit_psi4_final = pd.merge(dataset_iolitech_rdkit_psi4, cation_smiles, left_on=['cation'],
                                                     right_on=['name_cation'])

        # Export dataset_iolitech_rdkit_psi4_final as "dataset_iolitech_final.csv"
        dataset_iolitech_rdkit_psi4_final.to_csv(os.path.join(self.abs_path, "output", "dataset_iolitech_final.csv"),
                                                 index=False)


        return dataset_iolitech_rdkit_psi4_final

    # Prepare the data for training
    def machine_learning_data_prepare(self):
        # Verify the existence of the predecessor file, if it exists, then read it directly, if not, then get it through the function
        if os.path.exists(
                os.path.join(self.abs_path, self.abs_path, "output", "dataset_iolitech_final.csv")) == True:
            dataset = pd.read_csv(os.path.join(self.abs_path, self.abs_path, "output", "dataset_iolitech_final.csv"),
                                  index_col=None)
        elif os.path.exists(
                os.path.join(self.abs_path, "output", "dataset_iolitech_rdkit.csv")) == True and os.path.exists(
            os.path.join(self.abs_path, "output", "dataset_iolitech_final.csv")) == False:
            dataset = pd.read_csv(os.path.join(self.abs_path, "output", "dataset_iolitech_rdkit.csv"),
                                  index_col=None)
            dataset = self.psi_cal()
        elif os.path.exists(
                os.path.join(self.abs_path, "output", "dataset_iolitech_type.csv")) == True and os.path.exists(
            os.path.join(self.abs_path, "output", "dataset_iolitech_rdkit.csv")) == False and os.path.exists(
            os.path.join(self.abs_path, "output", "dataset_iolitech_final.csv")) == False:
            dataset = pd.read_csv(os.path.join(self.abs_path, "output", "dataset_iolitech_type.csv"), index_col=None)
            dataset = self.Descriptor()
            dataset = self.psi_cal()
        elif os.path.exists(
                os.path.join(self.abs_path, "output", "dataset_iolitech.csv")) == True and os.path.exists(
            os.path.join(self.abs_path, "output", "dataset_iolitech_type.csv")) == False and os.path.exists(
            os.path.join(self.abs_path, "output", "dataset_iolitech_rdkit.csv")) == False and os.path.exists(
            os.path.join(self.abs_path, "output", "dataset_iolitech_final.csv")) == False:
            dataset = pd.read_csv(os.path.join(self.abs_path, "output", "dataset_iolitech.csv"), index_col=None)
            dataset = self.data_processing()
            dataset = self.Descriptor()
            dataset = self.psi_cal()
        elif os.path.exists(
                os.path.join(self.abs_path, "output", "dataset_iolitech.csv")) == False and os.path.exists(
            os.path.join(self.abs_path, "output", "dataset_iolitech_type.csv")) == False and os.path.exists(
            os.path.join(self.abs_path, "output", "dataset_iolitech_rdkit.csv")) == False and os.path.exists(
            os.path.join(self.abs_path, "output", "dataset_iolitech_final.csv")) == False:
            dataset = self.dataCrawler()
            dataset = self.data_processing()
            dataset = self.Descriptor()
            dataset = self.psi_cal()

        #cation_dict = pd.read_csv(os.path.join(self.abs_path, "output", "cation_dict.csv"), index_col=None)
        #anion_dict = pd.read_csv(os.path.join(self.abs_path, "output", "anion_dict.csv"), index_col=None)

        # Merge "dataset","cation_dict" and "anion_dict" as "dataset"
        #dataset = pd.merge(dataset, cation_dict, on="cation", how='left')
        #dataset = pd.merge(dataset, anion_dict, on="anion", how='left')

        # 1 hartree = 27.21 eV/per
        dataset['V_al_cation'] = -dataset['HOMO_cation'] * 27.2
        dataset['V_al_anion'] = -dataset['HOMO_anion'] * 27.2
        dataset['V_al'] = dataset[['V_al_cation', 'V_al_anion']].min(axis=1)
        dataset['V_cl_cation'] = -dataset['LUMO_cation'] * 27.2
        dataset['V_cl_anion'] = -dataset['LUMO_anion'] * 27.2
        dataset['V_cl'] = dataset[['V_cl_cation', 'V_cl_anion']].max(axis=1)
        dataset['ECW_computed'] = dataset['V_al'] - dataset['V_cl']  # Calculate the ECW of ion pairs

        # Assign the type for cations and anions
        dataset = dataset.reset_index(drop=True)
        cation_kind = ['imidazolium', 'pyridinium', 'piperidinium', 'pyrrolidinium',
                       'ammonium', 'sulfonium', 'phosphonium', 'choline']

        anion_kind = ['imid', 'tetrafluoroborate', 'triflate',
                      'sulfate', 'dicyanamide', 'tricyanomethanide',
                       'thiocyanate', 'bromide', 'chloride', 'acetate',
                      'sulfonate', 'tosylate', 'formate',
                      'nitrate', 'phosph', 'decanoate', 'carbonate', 'cobaltate', 'hydroxide']
        cation_type = []
        anion_type = []
        for i in range(0, len(dataset)):
            if len(cation_type) != len(anion_type):
                print(dataset['cation'][i - 1], dataset['anion'][i - 1])
            for kind in cation_kind:
                dummy = len(cation_type)
                if kind in dataset['cation'][i]:
                    cation_type.append(kind)
                    break

            for kind in anion_kind:
                if kind in dataset['anion'][i]:
                    anion_type.append(kind)
                    break

        dataset['cation_type'] = cation_type
        dataset['anion_type'] = anion_type

        dataset_labeled = pd.read_csv(os.path.join(self.abs_path, "output", "dataset_iolitech_type.csv"),
                                      index_col=None)

        dataset_labeled['anion'] = dataset_labeled['anion'].str.lower()
        dataset_labeled['cation'] = dataset_labeled['cation'].str.lower()

        # Inner join dataset can be used for supervised learning
        # Left join dataset can be used for unsupervised learning, aggregated clustering
        dataset_comb_left = pd.merge(dataset, dataset_labeled, on=['anion', 'cation', 'cation_type', 'anion_type'],
                                     how='left')

        # Label the dataset
        state = []
        dataset_comb_left['melting_point'] = dataset_comb_left['melting_point'].fillna('unknown')

        # Define the is_number(s) function to convert the data type to float
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                pass
            return False

        # Compare the melting point with room temperature to assign different states of ionic liquids
        for i in range(0, len(dataset_comb_left)):
            if is_number(dataset_comb_left['melting_point'][i]):
                if float(dataset_comb_left['melting_point'][i]) <= 25:
                    state.append('liquid')
                else:
                    state.append('solid')
            elif dataset_comb_left['melting_point'][i] == 'unknown':
                state.append('unknown')
            elif dataset_comb_left['melting_point'][i] == "~RT":
                state.append('liquid')
            elif dataset_comb_left['melting_point'][i] in ["<RT\t", "<RT"]:
                state.append('liquid')
            elif dataset_comb_left['melting_point'][i] in [">RT\t", ">RT"]:
                state.append('solid')
            elif dataset_comb_left['melting_point'][i].split("°C")[0] in ['>270 ', '>300 ', '80-85 ', '75-80 ',
                                                                          '45-50 ', '52-53 ', '30-35 ', '60-70 ',
                                                                          '55-60 ', '>260 ', '>25', '>230 ',
                                                                          '>280 ',
                                                                          '40-45 ', '35-37 ']:
                state.append('solid')
            elif "°C" in dataset_comb_left['melting_point'][i]:
                if int(dataset_comb_left['melting_point'][i].split("°C")[0]) > 25:
                    state.append('solid')
                elif int(dataset_comb_left['melting_point'][i].split("°C")[0]) <= 25:
                    state.append('liquid')
            else:
                state.append('solid')

        dataset_comb_left['state'] = state

        """
        #Remove the outliers from the dataset
        def remove_outlier(dataset, feature):
            if "NumRadicalElectrons" not in feature:
                Q1 = dataset[feature].quantile(0.25)
                Q3 = dataset[feature].quantile(0.75)
                IQR = Q3 - Q1
                Lower_Whisker = Q1 - 1.5 * IQR
                Upper_Whisker = Q3 + 1.5 * IQR
                #print(Q1, Q3, IQR)
                #print(Lower_Whisker, Upper_Whisker)
                dataset = dataset[(dataset[feature] < Upper_Whisker) | (dataset[feature] > Lower_Whisker)]
            else:
                pass
            return dataset
        for feature in self.feature_list:
            #print(feature)
            dataset_comb_left = remove_outlier(dataset_comb_left, feature)
            #print(len(dataset_comb_left))
        """
        dataset_comb_left.to_csv(os.path.join(self.abs_path, "output", "dataset_comb.csv"), index=None)

        return dataset_comb_left

    # The Graphical Convolutional Neural Network based on PyTorch Geometrics and RDKit
    def gcnnModel(self):
        # The two layer GCN model
        class GCN(torch.nn.Module):
            def __init__(self):
                super(GCN, self).__init__()
                super().__init__()
                self.conv1 = GCNConv(46, 32)
                self.conv2 = GCNConv(32, 4)
                self.classifier = Linear(4, 1)

            def forward(self, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch
                emb = x
                x = self.conv1(x, edge_index)
                x = x.tanh()
                x = self.conv2(x, edge_index)
                x = x.tanh()

                # Apply a final (linear) classifier
                out = nn.global_mean_pool(x, batch)
                out = self.classifier(out)
                return emb, out

        ################################################################################################################
        # The credits of the following code blocks in gcnnModel should go to ############################################
        # https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/#
        ################################################################################################################

        # One hot encoding
        def one_hot_encoding(x, permitted_list):
            if x not in permitted_list:
                x = permitted_list[-1]
            binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
            return binary_encoding

        # Get the atom features
        def get_atom_features(atom,
                              use_chirality=True,
                              hydrogens_implicit=True):

            # define list of permitted atoms
            permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B']
            if hydrogens_implicit == False:
                permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

            # compute atom features
            atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
            n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
            formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
            hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()),
                                                      ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
            is_in_a_ring_enc = [int(atom.IsInRing())]
            is_aromatic_enc = [int(atom.GetIsAromatic())]
            atomic_mass_scaled = [atom.GetMass()]
            vdw_radius_scaled = [Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())]
            covalent_radius_scaled = [Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())]
            atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
            if use_chirality == True:
                chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()),
                                                      ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
                                                       "CHI_OTHER"])
                atom_feature_vector += chirality_type_enc
            if hydrogens_implicit == True:
                n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
                atom_feature_vector += n_hydrogens_enc
            return np.array(atom_feature_vector)

        # Get the bond features
        def get_bond_features(bond, use_stereochemistry=True):

            permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                            Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
            bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)

            bond_is_conj_enc = [int(bond.GetIsConjugated())]

            bond_is_in_ring_enc = [int(bond.IsInRing())]

            bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

            if use_stereochemistry == True:
                stereo_type_enc = one_hot_encoding(str(bond.GetStereo()),
                                                   ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
                bond_feature_vector += stereo_type_enc
            return np.array(bond_feature_vector)

        # Create the graph input by combing atom and bond features
        def create_pytorch_geometric_graph(x_smiles, y):
            data_list = []

            for (smiles, y_val) in zip(x_smiles, y):

                # convert SMILES to RDKit mol object
                mol = Chem.MolFromSmiles(smiles)
                # get feature dimensions
                n_nodes = mol.GetNumAtoms()
                n_edges = 2 * mol.GetNumBonds()
                unrelated_smiles = "O=O"
                unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
                n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
                n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))
                # construct node feature matrix X of shape (n_nodes, n_node_features)
                X = np.zeros((n_nodes, n_node_features))
                for atom in mol.GetAtoms():
                    X[atom.GetIdx(), :] = get_atom_features(atom)

                X = torch.tensor(X, dtype=torch.float)

                # construct edge index array E of shape (2, n_edges)
                (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
                torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
                torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
                E = torch.stack([torch_rows, torch_cols], dim=0)

                # construct edge feature array EF of shape (n_edges, n_edge_features)
                EF = np.zeros((n_edges, n_edge_features))

                for (k, (i, j)) in enumerate(zip(rows, cols)):
                    EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))

                EF = torch.tensor(EF, dtype=torch.float)

                # construct label tensor
                y_tensor = torch.tensor(np.array([y_val]), dtype=torch.float)

                # construct Pytorch Geometric data object and append to data list
                data_list.append(Data(x=X, edge_index=E, edge_attr=EF, y=y_tensor))
            return data_list

        # evaluate the model
        def evaluate_model(val_data_list, gnn_model):
            predictions, actuals = list(), list()
            dataloader = DataLoader(dataset=pred_data_list, batch_size=len(val_data_list))
            for i, batch in enumerate(val_data_list):
                # evaluate the model on the test set
                emb, yhat = gnn_model(batch)
                # retrieve numpy array
                yhat = yhat.detach().numpy()
                actual = batch.y.numpy()
                actual = actual.reshape((len(actual), 1))
                # round to class values
                yhat = yhat.round()
                # store
                predictions.append(yhat)
                actuals.append(actual)
            predictions, actuals = np.vstack(predictions), np.vstack(actuals)
            # calculate accuracy
            acc = accuracy_score(actuals, predictions)
            return acc

        # make a class prediction for one row of data
        def predict(pred_data_list, gnn_model):
            predictions = list()
            dataloader = DataLoader(dataset=pred_data_list, batch_size=len(pred_data_list))
            for k, batch in enumerate(dataloader):
                # make prediction
                emb, yhat = gnn_model(batch)
                # round to class values
                yhat = yhat.round()
                # retrieve numpy array
                yhat = yhat.detach().numpy()
                predictions.append(yhat)

            return predictions

        print("Prepare the dataset for gcnnModel")
        dataset = self.machine_learning_data_prepare()
        dataset = dataset[['pair_sml', 'state', 'pair']]
        train_dataset = dataset[dataset['state'] != 'unknown']
        train_dataset = train_dataset.sample(len(train_dataset), random_state=1)

        train_start = 0
        train_end = round(len(train_dataset) * 0.8)
        factor = pd.factorize(train_dataset['state'], sort=True)
        train_dataset['state'] = factor[0]

        train_x_smiles = list(train_dataset['pair_sml'].iloc[train_start:train_end])

        train_y = list(train_dataset['state'].iloc[train_start:train_end])

        val_x_smiles = list(train_dataset['pair_sml'].iloc[train_end:])

        val_y = list(train_dataset['state'].iloc[train_end:])

        # create list of molecular graph objects from list of SMILES x_smiles and list of labels y
        pred_dataset = dataset[dataset['state'] == 'unknown']
        pred_x_smiles = list(pred_dataset['pair_sml'])
        train_data_list = create_pytorch_geometric_graph(train_x_smiles, train_y)
        val_data_list = create_pytorch_geometric_graph(val_x_smiles, val_y)
        pred_data_list = create_pytorch_geometric_graph(pred_x_smiles, len(pred_x_smiles) * [0])
        # set model to training mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        gnn_model = GCN()
        # create dataloader for training
        dataloader = DataLoader(dataset=train_data_list, batch_size=2 ** 6)
        # define loss function
        loss_function = torch.nn.MSELoss()
        # define optimiser
        optimiser = torch.optim.Adam(gnn_model.parameters(), lr=1e-3)
        # canonical training loop for a Pytorch Geometric GNN model gnn_model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # loop over 250 training epochs
        epoch_accuracy = []
        for epoch in range(1000):
            # loop over minibatches for training
            for k, batch in enumerate(dataloader):
                # compute current value of loss function via forward pass
                # print(k)
                # print(len(batch.x))
                # print(len(batch.x[0]))
                # print(torch.tensor(batch.x))
                # data = batch.to(device)
                emb, output = gnn_model(batch)
                loss_function_value = loss_function(output[:, 0], torch.tensor(batch.y, dtype=torch.float32))
                # set past gradient to zero
                optimiser.zero_grad()
                # compute current gradient via backward pass
                loss_function_value.backward()
                # update model weights using gradient and optimisation method
                optimiser.step()
            acc = evaluate_model(val_data_list, gnn_model)
            epoch_accuracy.append(acc)

        # Evaluate the mode
        acc = evaluate_model(val_data_list, gnn_model)
        print(acc)
        # Predict the mode
        results = predict(pred_data_list, gnn_model)
        pred_dataset["state"] = results[0]
        gnn_liquid_results = pred_dataset[pred_dataset["state"] == 0]

        gnn_liquid_results.to_csv(os.path.join(self.abs_path, "output", "gnn_liquid_results.csv"), index=False)

    # This machineLearning() can be used for the regression and classification in the supervised learning.
    # method choices: "state_clf" or "conductivity_clf" or "conductivity_reg"
    def machineLearning(self, method):
        #Define function "calculate_vif" to check the of the multicollinearity of dataset variables
        def calculate_vif(df, features):
            vif, tolerance = {}, {}
            # all the features that you want to examine
            for feature in features:
                # extract all the other features you will regress against
                X = [f for f in features if f != feature]
                X, y = df[X], df[feature]
                # extract r-squared from the fit
                r2 = LinearRegression().fit(X, y).score(X, y)

                # calculate tolerance
                tolerance[feature] = 1 - r2
                # calculate VIF
                vif[feature] = 1 / (tolerance[feature])
            # return VIF DataFrame
            return pd.DataFrame({'VIF': vif, 'Tolerance': tolerance})

        #Define  function "machine_learning_data_processing" to prepare the dataset for traning and prediction
        def machine_learning_data_processing(dataset_comb_left, feature_list, label_feature):

            dataset_comb_left = dataset_comb_left.sample(n=len(dataset_comb_left))
            VIF_results = calculate_vif(dataset_comb_left, self.feature_list)
            VIF_results = VIF_results.sort_values(by = ["VIF"], ascending = False)
            VIF_results.to_csv(os.path.join(self.abs_path, "output", label_feature+ "VIF_results.csv"), index=True)
            #print(VIF_results[VIF_results["VIF"]>20])
            #print(VIF_results[VIF_results["VIF"]>20].index)
            #self.feature_list = list(VIF_results[VIF_results["VIF"]<50].index)

            # Predict the unknown state of the ionic liquid
            if label_feature == 'state':
                dataset_supervised = dataset_comb_left[dataset_comb_left[label_feature] != 'unknown']
                factor = pd.factorize(dataset_supervised[label_feature], sort=True)  # Encoding of text type features
                print(factor[1])
                dataset_supervised[label_feature] = factor[0]
                dataset_predict = dataset_comb_left[dataset_comb_left[label_feature] == 'unknown']

            # Predict ionic liquids of unknown conductivity
            elif label_feature == 'type':
                dataset_supervised = dataset_comb_left[dataset_comb_left[label_feature] != 2]
                dataset_predict = dataset_comb_left[dataset_comb_left[label_feature] == 2]

            # Predict ionic liquids of unknown ECW
            elif label_feature == 'ECW':
                dataset_supervised = dataset_comb_left[dataset_comb_left[label_feature] > 0]
                dataset_predict = dataset_comb_left[dataset_comb_left[label_feature] == 0]

            # When [label_feature] == 0, it means that label_feature is unknown and needs to be predicted
            else:
                dataset_supervised = dataset_comb_left[dataset_comb_left[label_feature] > 0]
                dataset_predict = dataset_comb_left[dataset_comb_left[label_feature] == 0]
                print("!!!!!!!!!dataset_predict!")
            print("Model data set size", len(dataset_supervised))
            print("Predict data set size", len(dataset_predict))



            dataTrain = dataset_supervised
            dataPred = dataset_predict

            train = dataTrain[self.feature_list]
            pred_data = dataPred[self.feature_list]

            # Normalize the train set and pred_data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(train)
            train = scaler.transform(train)
            pred_data = scaler.transform(pred_data)

            # Save the scaler
            scalerfile = self.abs_path + "/scaler/" + label_feature + '_scaler.sav'
            pickle.dump(scaler, open(scalerfile, 'wb'))

            train = DataFrame(train, columns=self.feature_list)
            pred_data = DataFrame(pred_data, columns=self.feature_list)

            # Build X and y
            X_train = train
            y_train = dataTrain[label_feature]

            X_pred = pred_data
            X_label = dataPred['pair']
            return X_train, y_train, X_pred, X_label

        # Define classifier functions "train_pred_clf" to classify the state of ionic liquids
        def train_pred_clf(X_train, y_train, X_pred, X_label, label_feature):


            # defining parameter range
            param_grid = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                          'kernel': ['rbf']}
            svm_base = SVC(kernel='rbf')
            svm_clf = GridSearchCV(svm_base, param_grid, refit=True, verbose=0, n_jobs=-1)

            # Parameter tuning for random forest model
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start=10, stop=200, num=5)]
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(2, 50, num=5)]
            # Minimum number of samples required to split a node
            min_samples_split = [1, 2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 5, 10]
            # Maximum number of leaf node
            max_leaf_nodes = [4, 10, 20, 50, None]
            # Create the random grid
            random_grid_rf = {'n_estimators': n_estimators,
                              'max_depth': max_depth,
                              'min_samples_split': min_samples_split,
                              'min_samples_leaf': min_samples_leaf,
                              'max_leaf_nodes': max_leaf_nodes}

            random_grid_xgb = {
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'n_estimators': n_estimators}

            clf_name = ['SVM', 'RF', 'XGB']

            rf_base = RandomForestClassifier()
            # Random search of parameters, using 3 fold cross validation,
            # search across 100 different combinations, and use all available cores
            rf_clf = GridSearchCV(rf_base, random_grid_rf, scoring='roc_auc', n_jobs=-1)
            xgb_base = XGBClassifier(eval_metric = 'logloss', use_label_encoder=False)
            xgb_clf = GridSearchCV(xgb_base, random_grid_xgb,  verbose=0, n_jobs=30, scoring='roc_auc')
            clf_list = [svm_clf, rf_clf, xgb_clf]
            results_df = []
            R_square = []
            feature_importance_dict = {}
            for i in range(0, len(clf_name)):
                clf = clf_list[i]

                print(clf)
                clf.fit(X_train, y_train)
                print("THe best parameter for " + clf_name[i], clf.best_params_)
                print('Best score:', clf.best_score_)
                #Replace the clf with using the best parameter
                clf = clf.best_estimator_


                # K-fold validation of the model
                kf = KFold(n_splits=5)
                score = cross_val_score(clf, X_train, y_train, cv=kf)

                print(clf.__class__.__name__, "Cross Validation Scores are {}".format(score))
                print("____________")
                print(clf.__class__.__name__, "Average Cross Validation score :{}".format(score.mean()))

                clf.fit(X_train, y_train)

                # print(a)
                # print("!!!!!1", clf.predict(a))
                # save the model to disk
                filename = self.abs_path + "/scaler/" + label_feature + '_xgb_model.sav'

                pickle.dump(clf, open(filename, 'wb'))
                loaded_model = pickle.load(open(filename, 'rb'))
                # print(loaded_model)
                # print(loaded_model.predict(a))

                # Obtain the accuracy of each classifier on the test set

                if i > 0:
                    feature_importances = pd.DataFrame(clf.feature_importances_,
                                                       index=X_train.columns,
                                                       columns=['importance']).sort_values('importance', ascending=True)

                    feature_importance_dict[clf_name[i]] = pd.Series(clf.feature_importances_,
                                                                     index=X_train.columns).sort_values(ascending=True)
                    feature_importance_clf_df = DataFrame(feature_importance_dict)
                    feature_importance_clf_df.to_csv(os.path.join(self.abs_path, "output",
                                                                  label_feature+"feature_importance_clf_df.csv"), index=True)

                predict = clf.predict(X_pred)
                results_df.append(predict)

            results = DataFrame({"pair": X_label,
                                 "SVM_predict": results_df[0], "RF_predict": results_df[1],
                                 "XGB_predict": results_df[2]})

            return results

        def data_regression(dataset_comb_left):
            #dataset = pd.read_csv(os.path.join(self.abs_path, "output", "dataset_comb.csv"), index_col=None)
            #dataset = dataset.fillna(0)
            #X_train, y_train, X_pred, X_label = machine_learning_data_processing(dataset, self.feature_list, "state")
            #results = train_pred_clf(X_train, y_train, X_pred, X_label, "state")
            dataset = dataset_comb_left
            dataset = dataset.fillna(0)
            dataset_reg = dataset
            results = pd.read_csv(os.path.join(self.abs_path, "output",
                                                          "results_state.csv"), index_col=None)
            gnn_liquid_results = pd.read_csv(os.path.join(self.abs_path, "output",
                                                          "gnn_liquid_results.csv"), index_col=None)
            liquid_IL = pd.merge(dataset_reg, results[
                (results["XGB_predict"] == 0) & (results["RF_predict"] == 0) & (results["SVM_predict"] == 0)]["pair"],
                                 on="pair")
            liquid_IL = pd.merge(liquid_IL, gnn_liquid_results[["pair_sml"]], on="pair_sml")
            liquid_IL_known = dataset[dataset["state"] == 'liquid']
            dataset_reg = pd.concat([liquid_IL, liquid_IL_known], axis=0)
            dataset_reg = dataset_reg.drop_duplicates()
            dataset_reg = dataset_reg.reset_index()

            label = []
            # Assign different grades depending on the known conductivity of the ILs
            for i in range(0, len(dataset_reg)):
                if dataset_reg["conductivity_norm"][i] == 0:
                    label.append(2)
                elif dataset_reg["conductivity_norm"][i] >= 5:
                    label.append(1)
                elif dataset_reg["conductivity_norm"][i] < 5:
                    label.append(0)

            dataset_reg['type'] = label

            return dataset_reg

        # Use to make predictions about the conductivity of ionic liquids
        # Define the function "train_pred_reg" to set the Support Vector Machine, Random Forest, Gradient boosting,
        # XGBoost and Voting models
        def train_pred_reg(X_train, y_train, X_pred, X_label, label_feature):
            reg_name = ['SVR', "RF", 'XGB']
            # Training regression

            # defining parameter range
            param_grid = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                          'kernel': ['rbf']}
            svm_base = SVR(kernel='rbf')
            svm_reg = GridSearchCV(svm_base, param_grid, refit=True, verbose=0, n_jobs=-1)

            # Parameter tuning for random forest model
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start=10, stop=200, num=5)]
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(2, 50, num=5)]
            # Minimum number of samples required to split a node
            min_samples_split = [1, 2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 5, 10]
            # Maximum number of leaf node
            max_leaf_nodes = [4, 10, 20, 50, None]
            # Create the random grid
            random_grid_rf = {'n_estimators': n_estimators,
                              'max_depth': max_depth,
                              'min_samples_split': min_samples_split,
                              'min_samples_leaf': min_samples_leaf,
                              'max_leaf_nodes': max_leaf_nodes}

            random_grid_xgb = {
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'n_estimators': n_estimators}

            rf_reg_base = RandomForestRegressor()
            rf_reg = GridSearchCV(rf_reg_base, random_grid_rf, refit=True, verbose=0, n_jobs=-1)
            xgb_reg_base = XGBRegressor()
            xgb_reg = GridSearchCV(xgb_reg_base, random_grid_xgb, refit=True, verbose=0, n_jobs=10)

            reg_list = [svm_reg, rf_reg, xgb_reg]
            results_df = []
            resutls_training_df = []
            resutls_testing_df = []
            MAE = []
            RMSE = []
            R_square = []
            feature_importance_dict = {}

            for i in range(0, len(reg_name)):
                reg = reg_list[i]

                print(reg)
                reg.fit(X_train, y_train)
                print("THe best parameter for " + reg_name[i], reg.best_params_)
                print('Best score:', reg.best_score_)
                #Replace the clf with using the best parameter
                reg = reg.best_estimator_


                # K-fold validation of the model
                kf = KFold(n_splits=5)
                score = cross_val_score(reg, X_train, y_train, cv=kf, scoring='r2')

                print(reg.__class__.__name__, "Cross Validation Scores are {}".format(score))
                print("____________")
                print(reg.__class__.__name__, "Average Cross Validation score :{}".format(score.mean()))

                reg.fit(X_train, y_train)
                if i >= 1 and i <= 2:
                    feature_importances = pd.DataFrame(reg.feature_importances_,
                                                       index=X_train.columns,
                                                       columns=['importance']).sort_values('importance', ascending=True)

                    feature_importance_dict[reg_name[i]] = pd.Series(reg.feature_importances_,
                                                                     index=X_train.columns).sort_values(ascending=True)

                predict = reg.predict(X_pred)

                results_df.append(predict)

            results = DataFrame({"pair": X_label,
                                 "SVM_predict": results_df[0], "RF_predict": results_df[1],
                                 "XGB_predict": results_df[2]})

            feature_importance_reg_df = DataFrame(feature_importance_dict)
            feature_importance_reg_df.to_csv(os.path.join(self.abs_path, "output", label_feature+"_feature_importance_reg_df.csv"),
                                             index=False)

            return results

        # Starting point of the ML steps
        if method == "state_clf":
            dataset = self.machine_learning_data_prepare()
            dataset = dataset.fillna(0)
            X_train, y_train, X_pred, X_label = machine_learning_data_processing(dataset, self.feature_list, "state")
            results_state = train_pred_clf(X_train, y_train, X_pred, X_label, "state")
            results_state.to_csv(os.path.join(self.abs_path, "output", "results_state.csv"), index=False)
            return results_state

        elif method == "conductivity_clf":
            dataset = self.machine_learning_data_prepare()
            dataset_clf = data_regression(dataset)
            X_train, y_train, X_pred, X_label = machine_learning_data_processing(pd.concat([dataset_clf], axis=0),
                                                                                 self.feature_list, "type")
            results_clf = train_pred_clf(X_train, y_train, X_pred, X_label, "type")
            results_clf.to_csv(os.path.join(self.abs_path, "output", "results_clf.csv"), index=False)
            return results_clf

        elif method == "conductivity_reg":
            dataset = self.machine_learning_data_prepare()
            dataset_reg = data_regression(dataset)
            X_train, y_train, X_pred, X_label = machine_learning_data_processing(pd.concat([dataset_reg], axis=0),
                                                                                 self.feature_list, "conductivity_norm")
            results_reg = train_pred_reg(X_train, y_train, X_pred, X_label, "conductivity_norm")
            results_reg.to_csv(os.path.join(self.abs_path, "output", "results_reg.csv"), index=False)
            return results_reg

        elif method == "ECW_reg":
            dataset = self.machine_learning_data_prepare()
            dataset_reg = data_regression(dataset)
            X_train, y_train, X_pred, X_label = machine_learning_data_processing(pd.concat([dataset_reg], axis=0),
                                                                                 self.feature_list, "ECW")
            results_reg = train_pred_reg(X_train, y_train, X_pred, X_label, "ECW")
            results_reg.to_csv(os.path.join(self.abs_path, "output", "results_ECW_reg.csv"), index=False)
            return results_reg

    # Screening of the ILs based on some thresholds.
    def screenIL(self):
        # Load the results from the regression and classification steps.
        dataset = self.machine_learning_data_prepare()
        results_state = pd.read_csv(os.path.join(self.abs_path, "output", "results_state.csv"), index_col=None)

        gnn_liquid_results = pd.read_csv(os.path.join(self.abs_path, "output", "gnn_liquid_results.csv"), index_col=None)
        gnn_liquid_results.columns = ["pair_sml", "gnn_predict", "pair"]
        print(results_state.head())
        results_state = pd.merge(results_state, gnn_liquid_results, on = "pair", how = "left")
        results_state["gnn_predict"] = results_state["gnn_predict"].fillna(1)
        print(results_state["gnn_predict"] )
        results_reg = pd.read_csv(os.path.join(self.abs_path, "output", "results_reg.csv"), index_col=None)
        results_clf = pd.read_csv(os.path.join(self.abs_path, "output", "results_clf.csv"), index_col=None)

        dataset_state = pd.merge(dataset, results_state, on="pair", how='outer')
        dataset_state_conduc = pd.merge(dataset_state, results_reg, on="pair", how='outer')
        results_final = pd.merge(dataset_state_conduc, results_clf, on="pair", how='outer')


        results_final.to_csv(os.path.join(self.abs_path, "output", "results_final.csv"), index=False)

        results_final["predicted_state"] = results_final["SVM_predict_x"] + results_final["RF_predict_x"] + \
                                           results_final["XGB_predict_x"] + results_final["gnn_predict"]

        results_final["predicted_conductivity"] = results_final["SVM_predict_y"] + results_final["RF_predict_y"] + \
                                                  results_final["XGB_predict_y"]
        results_final["predicted_conductivity"] = results_final["predicted_conductivity"] / 3
        results_final["predicted_conductivity_category"] = results_final["SVM_predict"] + results_final["RF_predict"] + \
                                                           results_final["XGB_predict"]


        # Filter based ont the liquid/solid state
        results_final_filtered_1 = results_final[
            (results_final["predicted_state"] == 0) | (results_final["predicted_state"].isna())]
        print(len(results_final_filtered_1))
        # Filter based on the category
        results_final_filtered_2 = results_final_filtered_1[
            (results_final_filtered_1["predicted_conductivity_category"].isna()) | (
                results_final_filtered_1["predicted_conductivity_category"] >= 2)]
        print(len(results_final_filtered_2))

        results_final_filtered_final = results_final_filtered_2[results_final_filtered_2["state"] != "solid"]
        print(len(results_final_filtered_final))
        results_final_filtered_final = results_final_filtered_final[results_final_filtered_final["V_al_anion"] != 0]
        print(len(results_final_filtered_final))
        results_final_filtered_final = results_final_filtered_final.fillna(0)
        results_final_filtered_final = results_final_filtered_final[
            (results_final_filtered_final["conductivity_norm"] >= 5) | (results_final_filtered_final["conductivity_norm"] == 0)]
        print(len(results_final_filtered_final))
        # Fiter based on the computed ECW values
        results_final_filtered_final = results_final_filtered_final[results_final_filtered_final["ECW_computed"] > 4]
        print(len(results_final_filtered_final))
        results_final_filtered_final["conductivity_combine"] = results_final_filtered_final["predicted_conductivity"] + results_final_filtered_final["conductivity_norm"]

        results_final_filtered_final.to_csv(os.path.join(self.abs_path, "output", "results_final_filtered_final.csv"),
                                            index=False)
        results_final.to_csv(os.path.join(self.abs_path, "output", "results_final.csv"), index=False)
        #Save for binding energy calculation
        cation_list = ["methylammonium","1-ethyl-3-methylimidazolium","ethyltributylphosphonium",
                       "1-butyl-1-methylpiperidinium","1-methyl-1-propylpyrrolidinium",
                       "diethylmethylsulfonium","1-ethyl-3-methylpyridinium"]
        anion_list = ["acetate","chloride","dicyanamide","bis(fluorosulfonyl)imide","nitrate","dihydrogen phosphate",
                      "ethyl sulfate","methanesulfonate","tetrafluoroborate","thiocyanate","tosylate",
                      "tricyanomethanide","triflate"]

        results_final_binding = results_final[(results_final["cation"].isin(cation_list)) & (results_final["anion"].isin(anion_list))]
        results_final_binding = results_final_binding.sort_values(["cation", "anion"])
        results_final_binding.to_csv(os.path.join(self.abs_path, "output", "results_final_binding.csv"), index=False)

    # The heriachical clustering is realized using heriaClustering()
    def heriaClustering(self):
        feature_list = ['HeavyAtomMolWt_cation', 'NumValenceElectrons_cation', 'volume_pair',
                        'MaxPartialCharge_anion', 'MaxAbsPartialCharge_anion',
                        'SpherocityIndex_anion', 'RadiusOfGyration_cation',
                        'SpherocityIndex_pair', 'InertialShapeFactor_anion',
                        'MaxAbsPartialCharge_cation', 'ECW_computed'
                        ]
        results_final_filtered_final = pd.read_csv(os.path.join(self.abs_path, "output",
                                                                "results_final_filtered_final.csv"), index_col=None)
        results_final = pd.read_csv(os.path.join(self.abs_path, "output", "results_final.csv"), index_col=None)

        results_final_clustering = results_final[
            (results_final["predicted_state"] == 0) | (results_final["predicted_state"].isna())]
        results_final_clustering = results_final_clustering[results_final_clustering["state"] != "solid"]
        results_final_clustering['abbr'] = results_final_clustering['cation'] + results_final_clustering['anion']

        def plot_dendrogram(model, **kwargs):
            # Create linkage matrix and then plot the dendrogram

            # create the counts of samples under each node
            counts = np.zeros(model.children_.shape[0])
            n_samples = len(model.labels_)
            for i, merge in enumerate(model.children_):
                current_count = 0
                for child_idx in merge:
                    if child_idx < n_samples:
                        current_count += 1  # leaf node
                    else:
                        current_count += counts[child_idx - n_samples]
                counts[i] = current_count

            linkage_matrix = np.column_stack([model.children_, model.distances_,
                                              counts]).astype(float)

            # Plot the corresponding dendrogram
            dendrogram_results = dendrogram(linkage_matrix, **kwargs)
            return dendrogram_results["ivl"]

        results_final_clustering = results_final_clustering.set_index('abbr')
        X = results_final_clustering[feature_list]
        fig, axes = plt.subplots(1, 1, figsize=(20, 20))
        X = X.dropna()
        labelList = X.index
        x = X.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)

        X_df = pd.DataFrame(x_scaled)
        # setting distance_threshold=0 ensures we compute the full tree.
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

        model = model.fit(X_df)

        plt.title('Hierarchical Clustering Dendrogram')
        # plot the top three levels of the dendrogram
        leaf_nodes = plot_dendrogram(model, truncate_mode='level', color_threshold=2,
                                     labels=list(labelList), leaf_font_size=5, leaf_rotation=90)
        fig.set_size_inches(40, 10)
        plt.grid(False)
        fig.patch.set_facecolor('white')
        plt.rcParams.update({'font.size': 1})
        plt.xlabel("")
        plt.savefig('Hierarchical Clustering Dendrogram' + ".png", dpi=600, bbox_inches='tight')
        leaf_nodes_df = pd.DataFrame({'abbr': leaf_nodes})
        results_final_filtered_final['star'] = 1
        print(len(results_final_filtered_final))
        results_final_filtered_final['abbr'] = results_final_filtered_final['cation'] + results_final_filtered_final[
            'anion']
        dataset_reorder_merge = pd.merge(leaf_nodes_df, results_final_filtered_final[['abbr', 'star']], on='abbr',
                                         how='left')
        # df = pd.DataFrame({'Conductivity': np.log(list(dataset_reorder['conductivity']))}, index=dataset_reorder.abbr)

        df = pd.DataFrame({'index': dataset_reorder_merge.abbr, 'star': list(dataset_reorder_merge['star'])})
        df["group"] = 1
        state_results = df.pivot_table(index="group", columns="index", values="star", dropna=False)
        state_results = state_results.reindex(list(df["index"]), axis="columns")
        # """
        sns.set_style("whitegrid", {'axes.grid': False})
        sns.set(rc={'figure.figsize': (50, 1)})
        sns.set(font_scale=0.1)
        ax = sns.heatmap(state_results, cmap='rocket_r')
        ax.set_ylabel("Star", fontsize=12)

        plt.savefig('Merge Hierarchical conductivity heamap' + ".png", dpi=600, bbox_inches='tight')
        plt.show()

    # Predict properties for new cation anion pairs.
    def modelPrediction(self, cation_smile, anion_smile):
        pair_smile = cation_smile + "." + anion_smile
        sml_list = [anion_smile, cation_smile, pair_smile]
        Rdkit_descriptor = []
        Psi4_descriptor = []
        for sml in sml_list:
            threedescriptor = self.cal3Ddescriptor(sml)
            Rdkit_descriptor += threedescriptor
            Moldescriptor = self.calMoldescriptor(sml)
            Rdkit_descriptor += Moldescriptor

        Psi4_descriptor += self.calMolEnergy(anion_smile)[0:-1]  # Total dipole not used
        Psi4_descriptor += self.calMolEnergy(cation_smile)[0:-1]

        test_data = Rdkit_descriptor + Psi4_descriptor
        test_data = np.array(test_data).reshape(1, -1)

        # Predict the state of the Cation and Anion pair.
        results_state = "The state is NA."
        results_type = "The conductivity is NA."
        # load the scaler
        scalerfile = self.abs_path + "/scaler/" + "state" + "_scaler.sav"
        scaler = pickle.load(open(scalerfile, 'rb'))
        test_scaled_set = scaler.transform(test_data)

        # load the model from disk
        modelfile = self.abs_path + "/scaler/" + "state" + "_xgb_model.sav"
        loaded_model = pickle.load(open(modelfile, 'rb'))
        results_state = loaded_model.predict(test_scaled_set)
        print("results", results_state)

        if results_state == [0]:
            results_state = "The ionic liquid pair is predicted to be liquid at room temperature."
            # load the scaler
            scalerfile = self.abs_path + "/scaler/" + "type" + "_scaler.sav"
            scaler = pickle.load(open(scalerfile, 'rb'))
            test_scaled_set = scaler.transform(test_data)

            # load the model from disk
            modelfile = self.abs_path + "/scaler/" + "type" + "_xgb_model.sav"
            loaded_model = pickle.load(open(modelfile, 'rb'))
            results_type = loaded_model.predict(test_scaled_set)
            if results_type == [0]:
                results_type = "The predicted conductivity is >  5 mS cm^(-1)."
            else:
                results_type = "The predicted conductivity is <= 5 mS cm^(-1)."
        else:
            results_state = "The ionic liquid pair is predicted to be solid at room temperature."
        return results_state, results_type

    # Validate and compare the predicted results to ILThermo Database
    def combineILthermo(self):
        # Data manipulation for ionic liquid ILthermo
        results_final = pd.read_csv(os.path.join(self.abs_path, "output", "results_final.csv"), index_col=None)
        ILthermo_dataset = pd.read_csv(os.path.join(self.abs_path, "output", "ionic_liquid_ILthermo_conductivcity.csv"))
        ILthermo_dataset["conductivity"] = ILthermo_dataset["Electrical conductivity, S/m"].str.split("±", expand=True)[
            0]
        ILthermo_dataset["conductivity"] = ILthermo_dataset["conductivity"].astype(float) * 10
        ILthermo_dataset["cation"] = ILthermo_dataset["Label"].str.split(" ", n=1, expand=True)[0]
        ILthermo_dataset["anion"] = ILthermo_dataset["Label"].str.split(" ", n=1, expand=True)[1]
        ILthermo_dataset["T_conductivity"] = ILthermo_dataset["Temperature, K"] - 273.15
        ILthermo_dataset["formula"] = ILthermo_dataset["Formula"]
        ILthermo_dataset["melting_point"] = ILthermo_dataset["Phase"]
        ILthermo = ILthermo_dataset[
            (ILthermo_dataset["Pressure, kPa"] > 99) & (ILthermo_dataset["Pressure, kPa"] < 105)]
        ILthermo = ILthermo.drop(["Pressure, kPa"], axis=1)
        ILthermo["label"] = ILthermo_dataset["Label"]
        print(len(results_final))
        print(len(ILthermo))
        dataset_combine = pd.merge(results_final, ILthermo[
            ["label", "cation", "anion", "T_conductivity", "formula", "conductivity", "melting_point"]],
                                   on=["cation", "anion"], how="inner")
        #Validation of ionic conductivity
        print(dataset_combine.columns)
        dataset_combine["avg_predict"] = (dataset_combine["XGB_predict_y"]+ dataset_combine["RF_predict_y"]+ dataset_combine["SVM_predict_y"])/3
        conductivity_valid = dataset_combine[
            (dataset_combine["T_conductivity_y"] == 25) & (dataset_combine["avg_predict"] > 0)]
        print(conductivity_valid)
        conductivity_valid = conductivity_valid.groupby("label_y", as_index = False)[["conductivity_y", "avg_predict","XGB_predict_y", "RF_predict_y", "SVM_predict_y"]].mean()
        conductivity_valid = conductivity_valid.sort_values("conductivity_y")
        print(conductivity_valid[["conductivity_y", "avg_predict", "XGB_predict_y", "RF_predict_y", "SVM_predict_y"]])
        print("Conductivity r2",
              r2_score(conductivity_valid["conductivity_y"],conductivity_valid["XGB_predict_y"]))
        print("Conductivity mean square error",
              mean_squared_error(conductivity_valid["conductivity_y"],conductivity_valid["XGB_predict_y"]))
        print("Conductivity mean absolue error",
              mean_absolute_error(conductivity_valid["conductivity_y"],conductivity_valid["XGB_predict_y"]))

        #Validation of ECW
        ECW_valid = results_final[results_final["ECW"] > 0][["label", "cation", "anion", "cation_type", "anion_type","ECW", "ECW_computed"]]
        print("ECW r2",
              r2_score(ECW_valid["ECW"],ECW_valid["ECW_computed"]))
        print("ECW mean square error",
              mean_squared_error(ECW_valid["ECW"],ECW_valid["ECW_computed"]))
        print("ECW mean absolute error",
              mean_absolute_error(ECW_valid["ECW"],ECW_valid["ECW_computed"]))

        conductivity_valid.to_csv(os.path.join(self.abs_path, "output", "conductivity_valid.csv"), index=False)
        ECW_valid.to_csv(os.path.join(self.abs_path, "output", "ECW_valid.csv"), index=False)
        dataset_combine.to_csv(os.path.join(self.abs_path, "output", "dataset_combine_ILthermo.csv"), index=False)





