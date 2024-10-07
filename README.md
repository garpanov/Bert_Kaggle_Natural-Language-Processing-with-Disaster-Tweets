Text Classification with BERT
This project implements a text classification model using BERT (Bidirectional Encoder Representations from Transformers). The model is designed to classify text data from a CSV file based on predefined labels.

Requirements
To run this code, you need the following libraries:

pandas
scikit-learn
torch
transformers
re
You can install the required libraries using pip:
pip install pandas scikit-learn torch transformers

Dataset
The code expects the following CSV files in the same directory:

train.csv: Contains the training data with at least two columns:

text: The text data to be classified.
target: The corresponding labels for the text data.
test.csv: Contains the test data with a text column similar to the training data.

sample_submission.csv: A sample submission file to structure the output.

Code Overview
Data Preprocessing:

The code normalizes the text by removing URLs and converting it to lowercase.
It further processes the text to remove mentions (@) and hashtags (#).
Tokenization:

The processed text is tokenized using the BERT tokenizer, padding and truncating to ensure consistent input sizes for the model.
Data Loading:

The processed data is converted into PyTorch tensors and loaded into a DataLoader for batching.
Model Initialization:

A BERT model for sequence classification is loaded with the specified number of labels (in this case, 2).
Training:

The model is trained for 3 epochs, and loss is printed every 10 batches.
Prediction:

The test dataset is processed similarly to the training dataset.
Predictions are made using the trained model, and probabilities are calculated using the softmax function.
Output:

The predicted classes are saved into a new column in the sample submission file, which is then exported as submission_1.csv.
The generated submission file is automatically downloaded.
Usage
Place your CSV files (train.csv, test.csv, sample_submission.csv) in the project directory.
Run the script to preprocess the data, train the BERT model, and generate predictions.
The output file submission_1.csv will be created in the same directory.
Note
Make sure you have access to a compatible GPU if you plan to run this code efficiently, as training BERT models can be resource-intensive.
