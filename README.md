This implementation is for the publication of `Quantizing Large-Language Models for Predicting Flaky Tests`.

cd src
```shell
bash flaky_test_categorization.sh IDoFT_2Cat "train"
```

```shell
bash flaky_test_categorization.sh IDoFT_2Cat "traditional_ml" "KNN"
```


```shell
bash flaky_test_categorization.sh IDoFT_6Cat "train"  
```

```shell
bash flaky_test_categorization.sh IDoFT_6Cat "traditional_ml" "KNN"
```

```shell
bash per_project_prediction.sh IDoFT "train"
```
```shell
bash per_project_prediction.sh IDoFT "test" "BERT"
```


All models are available in the following link
https://utexas.box.com/s/gdnwo6i18uhfro4xbfnmbohr58nlx0ho

