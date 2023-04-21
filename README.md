# ILP
This repository is the place where you can find the data and ML codes for the published manuscript with title of "Machine learning-guided discovery of ionic polymer electrolytes for lithium metal batteries"

1.  Document Introduction 
  1.	output folder：Used to store the output files for the ILP Class.
  2.	__init__.py：Source code of ILP Class.
  3.    manuscript_figure_data folder: all the raw data to generate figures in the manuscript. 
  
2. ILP Class Introduction
1.	dataCrawler(parser="html.parser",url="https://iolitec.de/en/products/list")
Default parser is "html.parser" and the target URL is https://iolitec.de/en/products/list. By using this method, the properties and chemical formula of the ion pairs in the URL can be crawled and cleaned. The final data will be exported as "dataset_iolitech.csv".
Note: The method takes a bit long time to finish scrapping the target information, which should be finished within 40 mins.
2.	dataProcessing():
This method processes the crawled data, including the ion types and the ionic conductivity, the viscosity and the density of the ionic liquids. The properties are all normalized to 25°C. The result is exported as "dataset_iolitech_type.csv".
3.	despGenerator() :
This method is used to calculate the molecular descriptors of cations and anions and ion pairs, and the final result is stored as "dataset_iolitech_rdkit.csv".
4.	psiCal() : 
This method calculates the atomic energy, HOMO, LUMO, atomic coordinates and dipole moment of cations, anions and ion pairs. The final result is stored as "dataset_iolitech_final.csv".
5.	gcnnModel():
	This method employs the graph convolutional neural network to classify the solid/liquid state of the ion pairs. The credits of this block should go to https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/.
6.	machineLearning(method)：
method = “state_clf” OR “conductivity_reg” OR “conductivity_clf”
Three main methods have been employed in this machine learning workflow, including “state_clf”, “conductivity_reg”, “conductivity_clf”. 
method = “state_clf”. This learner is a classifier that combines support vector machines, random forests, and XGBoost algorithms to classify the ion pairs of unknown physical state as solid or liquid. The result is stored as "results.csv".
method = “conductivity_reg”. The learner is a regression algorithm that predicts the ionic conductivity using support vector machine, random forest, and XGBoost regression algorithms for ionic liquids with unknown conductivity. The result is stored as "results_reg.csv".
method = “conductivity_clf”. The learner is a classifier that uses support vector machines, random forests, and XGBoost regression algorithms to classify the conductivity of ionic liquids with unknown conductivity into two categories: ionic conductivity >= 5 mS cm-1 or < 5m S cm-1. The result is stored as "results_clf.csv".
7.	screenIL():
	This method sets the thresholds to filter the IL based on the regression and classification results. The results are stored as "results_final.csv" and "results_final_filtered_final.csv" in the input and output file folders.
8.	heriaClustering():
	This method provides the execution and visualization of the dendrogram clustering. 
9.	combineILThermo():
This method compares the predicted results to ILThermo Database.
10.	modelPrediction():
This function can be used to predict new IL properties based on saved models. 
3. Software and library version:
1.	RDKit: 2022.03.05
2.	PyTorch: 1.13.0
3.	PyTorch Geometric: 2.2.0
4.	Psi4: 1.7a1.dev44


4. Sample code to use the ILP class
5. 

Import ILP

1.m = ILP()

2.m.dataCrawler()

3.m.dataProcessing()

4.m.despGenerator()

5.m.psiCal("scf/6-311g**")

6.m.gcnnModel()

7.m.machineLearning("state_clf")

8.m.machineLearning("conductivity_clf")

9.m.machineLearning("conductivity_reg")

10.m.screenIL()

11.m.combineILthermo()

