# cs5293sp22-project3

### Author : Pradipkumar Rajasekaran

Email: pradipkumar.rajasekaran-1@ou.edu

__Requirements to run__

GCP Instance Type (Minimum) : e2-medium, Memory: 4GB

__Running the Code__

1. Clone the repository- git clone https://github.com/CurSpace/cs5293sp22-project3.git
2. Navigate into the repo folder(cs5293sp22-project3)
3. Install all requirements using - pipenv install
4. From the repo folder run - pipenv run python -m pytest
5. To run the code - pipenv run python project3.py

__Example of running the code__

- Run the program by:

```
   pipenv run python project3.py
```

   __Sample Output:__
   
 ```
[('sadako', 70), ('christopher walken', 34), ('jackie chan', 30), ('donald sutherland', 21), ('von trier', 21), ('stanley', 18), ('roy scheider', 13), ('louis jouvet', 9), ('will smith', 9), ('brosnan', 8), ('ricky', 8), ('robin williams', 8), ('aidan quinn', 7), ('deepa mehta', 7), ('hitchcock', 7), ('molly ringwald', 7), ('corey feldman', 6), ('john', 6), ('walter matthau', 6), ('alice', 5)]
Precision:  0.016536419728502248 Recall: 0.01944005616991399 F1 Score: 0.01637915561691233
 ```
 
 __Description of Functions:__
 
 1. preprocess(review) 
                     - Takes the review as input and cleanes it but ignoring elements that are not letters or numbers and lowers all characters then it
                       it performs lemmatization on the review string.
                     - Then, the data frame is broken into lists(ingredient_list, cuisine_list, id_list).
                     - And returns the lists.

 2. splitData(data,flag)
                     - Reads the tsv updated tsv from https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv
                     - Creates a dataframe with column headings ['gitId','dataType','label','review']
                     - And returns train,valid and test sets.

 3. extractFeatures(train,valid,test) 
                     - Takes train,test and valid and stores X_train,X_valid,X_test,y_train,y_valid,y_test
                     - Removes stop words and vectorizes using count vecortizer to extract features.
                     - The valid and test data are vectorized using the train vocabulary
                     - Returns X_train,X_valid,X_test,y_train,y_valid,y_test

 4. training(X_train,y_train,X_test):
                     - Trains and fits to data to multinomial naive bayes classifier. 
                     - Uses grid search to select hyperparameter alpha.
                     - Returns the predictions on the test data.

 5. metrics(y_test,predictions):
                     - Takes y_test and predicions.
                     - Returns precision, recall and accuracy.

__Description of Test Functions:__

1. test_data(): 
                     - reads unredactor.tsv and stores it to be used by the other test functions.
 
2. test_splitDdata(test_data):
                     - gets the test data from test_data() and tests splitData() by checking the shape of the returned objects.
                                        
3. extractFeatures(test_data):
                     - tests the extractFeatures() by checking the shape of the returned objects.

4. test_training(test_data):
                     - test the training() by checking if len(predictions) == len of the  test data.

5. test_metrics(test_data):
                     - tests mertics() checking if the values for precision, recall and F1-score lie between 0 and 1.
 
__Bugs & Assumptions__

1. The data is \t delimited and must contain 4 rows.
  
2. The results are not perfect they are resonable. Gives top 20 names and scores. 

3. Bad rows may be skipped.

4. When reading the data into data frame the 1st row was header so had to push the headings down and add column headings.

5. The input data is acquired from https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv 


__Tested feature extraction methods__

1. Used dictionary vecorizer and considered len of redacted section, len of review, if the name was one or 2 words etc as features. However this yeilded poor results.
2. Used union of count and tfidf this also did not yeild satisfactory results. 
3. Used NLTK’s Pre-Trained Sentiment Analyzer to classify the reviews and positive or negative and use them as one of the features ,this also did not yeild any good resuls.
4. Then tried count and tfidf seperately the count vectorizer gave the best result.

__Tested classification models and methods__

1. Random forest classifier - Yielded poor results
2. mlp classifier - results were close to multinomial naive bayes but requires more training time
3. stochastic gredient descent - Yielded poor results.
4. Support Vector Machines - results were not better than multinomial naive bayes.
5. Multinomial Naive Bayes  - yielded the best results. Tried to create a pretrained model by training on all the 25000 reviews in the test data and save a pkl file
                              but the pkl file was 4gb. Had issues uploading to github and cloning will probably would take time. Therefore, just used the sklearn
                              MultinomialNB().
