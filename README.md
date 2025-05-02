This project manipulates a fasta file with metagenomic sequence data from the human gut into the csv that will be uploaded shortly. 
It uses k-mers to train an imbalanced random forest (because there's a lot of class imbalance in the dataset) and returns a 
text file with the results of the model. Note that this script will NOT run as is because there are local filepaths.
