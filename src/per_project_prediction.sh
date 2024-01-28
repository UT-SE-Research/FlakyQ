#Flakify's flaky test prediction using per-project validation
#dataset=$1
dataset_name=$(echo $1 | cut -d'_' -f1)
data_path="../dataset"
results_path="../flaky-test-categorization_per_project"
dataset_file="${data_path}/${dataset_name}/${dataset_name}_dataset.csv" 
model_weights="${results_path}/per_project_model_weights_on_${dataset}_dataset"
results_file="${results_path}/sklearn_predictor/per_project_validation_results_on_${dataset}_dataset"

if [[ ! -d "${results_path}/sklearn_predictor" ]]; then
    mkdir -p "${results_path}/sklearn_predictor"
fi
#Flakify_cross_validation_results_on_IDoFT_dataset_Flakify.csv
if [[ $2 == "train" ]]; then
    python3 -W ignore Bert_train_per_project.py $dataset_file $model_weights "${results_file}.csv" $1

elif [[ $2 == "test" ]]; then
    if [[ $3 == "traditional_ml" ]]; then
        python3 -W ignore Testing_per_project.py $dataset_file $model_weights "${results_file}_$4_inference.csv" "$4" $1  # Only for testing
    else #BERT
        python3 -W ignore Testing_per_project.py $dataset_file $model_weights "${results_file}_$4_inference.csv" "BERT" $1  # Only for testing
    fi
elif [[ $2 == "explain" ]]; then
    python3 explainable.py $dataset_file $model_weights "${results_file}.csv" 
fi


