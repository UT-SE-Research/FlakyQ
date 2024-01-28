#!/bin/bash

##For 2nd Part (Categorization-6 category-[flaky categories]) -For Paper Table-II and Table-III (for multiple category)
###bash flaky_test_categorization.sh IDoFT_6Cat "train"  
bash flaky_test_categorization.sh IDoFT_6Cat "BERT"  
bash flaky_test_categorization.sh IDoFT_6Cat "traditional_ml" "KNN"
bash flaky_test_categorization.sh IDoFT_6Cat "traditional_ml" "MLP"
bash flaky_test_categorization.sh IDoFT_6Cat "traditional_ml" "RF"
bash flaky_test_categorization.sh IDoFT_6Cat "traditional_ml" "SVC"
bash flaky_test_categorization.sh IDoFT_6Cat "traditional_ml" "LR"
#
####bash flaky_test_categorization.sh IDoFT_2Cat "train"  
###For 2nd Part (Categorization-2 category-[flaky,non-flaky]) -For Paper Table-II
bash flaky_test_categorization.sh IDoFT_2Cat "BERT"  
bash flaky_test_categorization.sh IDoFT_2Cat "traditional_ml" "KNN"
bash flaky_test_categorization.sh IDoFT_2Cat "traditional_ml" "MLP"
bash flaky_test_categorization.sh IDoFT_2Cat "traditional_ml" "RF"
bash flaky_test_categorization.sh IDoFT_2Cat "traditional_ml" "SVC"
bash flaky_test_categorization.sh IDoFT_2Cat "traditional_ml" "LR"
#
##For 2nd Part (Categorization-5 category-[from flakicat])  -For Paper Table-VII
###bash flaky_test_categorization.sh Flakicat "train" # on Flakicat data
bash flaky_test_categorization.sh Flakicat "BERT"
bash flaky_test_categorization.sh Flakicat "KNN"
bash flaky_test_categorization.sh Flakicat "MLP"
bash flaky_test_categorization.sh Flakicat "RF"
bash flaky_test_categorization.sh Flakicat "SVC"
bash flaky_test_categorization.sh Flakicat "LR"
bash flaky_test_categorization.sh Flakicat "test"
##
##For 3rd Part(Per-Project evaluation) # for 6 category -For Paper Table-V
###bash per_project_prediction.sh IDoFT_6Cat "train" 
bash per_project_prediction.sh IDoFT_6Cat "test" "BERT"
bash per_project_prediction.sh IDoFT_6Cat "test" "traditional_ml" "KNN"
bash per_project_prediction.sh IDoFT_6Cat "test" "traditional_ml" "MLP"
bash per_project_prediction.sh IDoFT_6Cat "test" "traditional_ml" "RF"
bash per_project_prediction.sh IDoFT_6Cat "test" "traditional_ml" "SVC"
bash per_project_prediction.sh IDoFT_6Cat "test" "traditional_ml" "LR"

##For 3rd Part(Per-Project evaluation) # for 2 category  -For Paper Table-VI
###bash per_project_prediction.sh IDoFT_2Cat "train"
bash per_project_prediction.sh IDoFT_2Cat "test" "BERT"
bash per_project_prediction.sh IDoFT_2Cat "test" "traditional_ml" "KNN"
bash per_project_prediction.sh IDoFT_2Cat "test" "traditional_ml" "MLP"
bash per_project_prediction.sh IDoFT_6Cat "test" "traditional_ml" "RF"
bash per_project_prediction.sh IDoFT_2Cat "test" "traditional_ml" "SVC"
bash per_project_prediction.sh IDoFT_2Cat "test" "traditional_ml" "LR"
