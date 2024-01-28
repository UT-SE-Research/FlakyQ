dataset_name=$(echo $1 | cut -d'_' -f1)
data_path="../dataset"
results_path="../flaky-test-categorization"
if [[ $1 == "IDoFT_2Cat" ]]; then #because the IDOFT_dataset.csv only contains tests which are flaky
    dataset_file="${data_path}/${dataset_name}/Flakify_${dataset_name}_dataset.csv" 
else
    dataset_file="${data_path}/${dataset_name}/${dataset_name}_dataset.csv" 
fi
model_weights="${results_path}/model_weights_on_${1}_dataset" # This one is needed to save the best model
results_file="${results_path}/sklearn_predictor/results_on_${1}_dataset"
#processed_data_with_vocab_file=base_path+"/data/processed_data_with_vocabulary_per_test.csv"

if [[ ! -d "${results_path}/sklearn_predictor" ]]; then
    mkdir -p "${results_path}/sklearn_predictor"
fi
#Flakify_cross_validation_results_on_IDoFT_dataset_Flakify.csv
if [[ $2 == "train" ]]; then
   if [[ $1 == "Flakicat" ]]; then 
        if [[ $3 == "Flakify-Org" ]]; then
            python Bert_train_flakify_original_categorization.py $dataset_file $model_weights "${results_file}_FlakiCat_BERT.csv"  "Flakicat_Categorization-Data" "BERT-Flakicat"
        else
           python3 Bert_train_categorization.py $dataset_file $model_weights "${results_file}_FlakiCat_BERT.csv"  "Flakicat_Categorization-Data" "BERT-Flakicat"
           #python3 Bert_train_categorization_codet5.py $dataset_file $model_weights "${results_file}_FlakiCat_BERT.csv"  "Flakicat_Categorization-Data" "BERT-Flakicat"
        fi
   elif [[ $1 == "IDoFT_6Cat" ]]; then # This one will be for 6 category
        if [[ $3 == "Flakify-Org" ]]; then
            echo "HIIIIIIIIIIIIIIIIIIIII"
            python Bert_train_flakify_original_categorization.py $dataset_file $model_weights "${results_file}_BERT.csv" "IDOFT_6Category-Data" "BERT-$1"
            exit
        else
            python3 Bert_train_categorization.py $dataset_file $model_weights "${results_file}_BERT.csv" "IDOFT_6Category-Data" "BERT-$1"
        fi

   else # This one will be for 2 category (flaky vs non-flaky)
       python3 Bert_train_categorization.py $dataset_file $model_weights "${results_file}_BERT.csv" "IDOFT_2Category-Data" "BERT-$1"
   fi

else # $2=test
    if [[ $1 == "Flakicat" ]]; then 
        if [[ $3 == "Flakify-Org" ]]; then
            python Testing_original_flakify.py $dataset_file $model_weights "${results_file}_$2.csv" "$2-Flakicat" "Flakicat_Categorization-Data"
        else
            python3 Testing_bert_flaky_categorization.py $dataset_file $model_weights "${results_file}_$2.csv" "$2-Flakicat" "Flakicat_Categorization-Data"  # $5 needed to load data
           fi
    elif [[ $2 == "traditional_ml" ]]; then
        echo "Traditional ml"
        if [[ $1 == "IDoFT_6Cat" ]]; then
            python3 Testing_bert_flaky_categorization.py $dataset_file $model_weights "${results_file}_$3.csv" "$3-$1" "IDOFT_6Category-Data"  # $5 needed to load data
        elif [[ $1 == "IDoFT_2Cat" ]]; then
            python3 Testing_bert_flaky_categorization.py $dataset_file $model_weights "${results_file}_$3.csv" "$3-$1" "IDOFT_2Category-Data"  # $5 needed to load data
        fi
    else # Only For BERT, inference, $2="BERT"
        if [[ $1 == "IDoFT_6Cat" ]]; then
            if [[ $3 == "Flakify-Org" ]]; then 
                echo "Flakify-Org IDOFT 6 cat"
                python Testing_original_flakify.py $dataset_file $model_weights "${results_file}_$2_inference.csv" "$2-$1" "IDOFT_6Category-Data"
            else
                echo "calling bert_inference $2"
    	        #python3 Bert_inference_flaky_categorization.py $dataset_file $model_weights "${results_file}_$2_inference.csv" "$2-$1" "IDOFT_6Category-Data"  # $5 needed to load data
                python3 Testing_bert_flaky_categorization.py $dataset_file $model_weights "${results_file}_$2.csv" "$2-$1" "IDOFT_6Category-Data"
            fi
        elif [[ $1 == "IDoFT_2Cat" ]]; then
    	    echo "I AM HERE"A
            python3 Testing_bert_flaky_categorization.py $dataset_file $model_weights "${results_file}_$2.csv" "$2-$1" "IDOFT_2Category-Data"
	        #python3 Bert_inference_flaky_categorization.py $dataset_file $model_weights "${results_file}_$2_inference.csv" "$2-$1" "IDOFT_2Category-Data"  # $5 needed to load data
	    fi
    fi
fi
