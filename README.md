## Tensorflow code of MobileNet + CoordConv based Rectification with TFLite and Pruning support

This code implements a mobilenet based model with coord-conv regression layers to find key-points of ID cards so that they can be rectified. Below are the files and their discriptions. 

* **runAll** script
	`Runs the whole processing from training a specified model and then testing the model on the dataset in tflite form which will be served on mobile. The parameters fed are as below`
	* Name with which the model is saved.
	* Path where dataset is stored in format specified below
	* Number of epochs
	`It does pruning by default and basically runs the following three files in this order, testing for each saved epoch for a single training session`
* **main**
	`The script to train a new model and save the model as .pb. Guide on its options can be seen via the --help option`
* **loadSave** 
	`load a .pb model and then save a .tflite model`
* **runTFLITE**
	`Test a .tflite model`

The dataset should be stored in this format if train/val/test data are in a single dataframe:-
* Folder containing files
	* labels.csv `All the labels associated with the data in a database with the file names`
	* images `A folder containing all the images`

Otherwise separate folders name `test`, `train` and `val` 