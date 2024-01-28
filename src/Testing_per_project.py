import time
import sys
import gc
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm, tree
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import sklearn.metrics as metrics
import torch.nn.utils.prune as prune
import torch.quantization
import torch.quantization as quantization
from torch.quantization import fuse_modules

import seaborn as sns
import matplotlib.pyplot as plt

def calculate_filtered_avg(arr, th=-1000):
    filtered_arr = [x for x in arr  if x != th]
    avg = sum(filtered_arr) / len(filtered_arr)
    return avg

def compute_metrics(test_y,preds):
    
    total_tp=0
    total_fp=0
    total_fn=0
    total_tn=0
    accuracies=[]
    f_scores=[]
    precisions=[]
    recalls=[]
    print(test_y)
    print(preds)
    for category in range(6):
        y=np.where(test_y==category,1,0) #taking y only for the specific category
        print(y)
        #print(test_y)
        p=np.where(preds==category,1,0)
        print("P====:w,"+str(category))
        print(p)
        n_p=np.sum(y==1) #positive
        n_n=np.sum(y==0) #negative
        
        cm = confusion_matrix(y, p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        print(cm.ravel())
        if (tp == 0.0):
            precision=0.0
            recall=0.0
            f_score=0.0
        else:
            precision=tp/(tp+fp)
            recall=tp/(tp+fn)
            f_score=2*(precision*recall)/(precision+recall)
        recalls.append(recall)
        precisions.append(precision)
        f_scores.append(f_score)
        total_tp +=tp
        total_tn +=tn
        total_fp +=fp
        total_fn +=fn

    accuracy=(total_tp+total_tn)/len(test_y)
    return total_tp, total_tn, total_fp, total_fn, precisions, recalls, f_scores, accuracy


# setting the seed for reproducibility
def set_deterministic(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True 

# specify GPU
device = torch.device("cuda")
#device = torch.device("cpu")

#reading the parameters 

dataset_path = sys.argv[1]
model_weights_path = sys.argv[2]
results_file = sys.argv[3]
ml_technique = sys.argv[4]
dataset_type = sys.argv[5]
#ml_technique=technique.split("-")[0]
#which_dataset=technique.split("-")[1]

df = pd.read_csv(dataset_path)
#input_data = df['full_code'] # use the 'full_code' column to run Flakify using the full code instead of pre-processed code
#target_data = df['category']
df.head()
# get project names

#project_name=df['project'].unique()

#project_name=['dubbo','hadoop','nifi','junit-quickcheck','ormlite-core','admiral','wildfly','Mapper','fastjson','typescript-generator','Chronicle-Wire','Java-WebSocket','biojava','spring-boot','hbase','visualee','adyen-java-api-library','innodb-java-reader','hive','spring-hateoas','DataflowTemplates','esper','spring-data-r2dbc','openhtmltopdf','nacos','mockserver','riptide']
project_name=['Chronicle-Wire', 'DataflowTemplates', 'Java-WebSocket', 'Mapper', 'admiral', 'adyen-java-api-library', 'biojava', 'dubbo', 'esper', 'fastjson', 'hadoop', 'hbase', 'hive', 'innodb-java-reader', 'junit-quickcheck', 'mockserver', 'nacos', 'nifi', 'openhtmltopdf', 'ormlite-core', 'riptide', 'spring-boot', 'spring-data-r2dbc', 'spring-hateoas', 'typescript-generator', 'visualee', 'wildfly']
#'biojava', 'dubbo', 'fastjson','hadoop', 'hbase','junit-quickcheck', 'nifi',
#project_name=['mockserver', 'nacos', 'openhtmltopdf', 'ormlite-core', 'riptide', 'spring-boot', 'spring-data-r2dbc', 'spring-hateoas', 'typescript-generator', 'visualee', 'wildfly']
# balance dataset
'''def sampling(X_train, y_train, X_valid, y_valid):
    
    oversampling = RandomOverSampler(
        sampling_strategy='minority', random_state=49)
    x_train = X_train.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)
    x_val = X_valid.values.reshape(-1, 1)
    y_val = y_valid.values.reshape(-1, 1)
    
    x_train, y_train = oversampling.fit_resample(x_train, y_train)
    x_val, y_val = oversampling.fit_resample(x_val, y_val)
    x_train = pd.Series(x_train.ravel())
    y_train = pd.Series(y_train.ravel())
    x_val = pd.Series(x_val.ravel())
    y_val = pd.Series(y_val.ravel())

    return x_train, y_train, x_val, y_val'''


# defining CodeBERT model
model_name = "microsoft/codebert-base"
model_config = AutoConfig.from_pretrained(model_name, return_dict=False, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
auto_model = AutoModel.from_pretrained(model_name, config=model_config)

# converting code into tokens and then vector representation
def tokenize_data(train_text, val_text, test_text):
    max_length = 1024
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length=max_length,
        pad_to_max_length=True,
        truncation=True)

    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length=max_length,
        pad_to_max_length=True,
        truncation=True)

    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length=max_length,
        pad_to_max_length=True,
        truncation=True)
    return tokens_train, tokens_val, tokens_test


# converting vector representation to tensors
def text_to_tensors(tokens_train, tokens_val, tokens_test):
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])

    return train_seq, train_mask, val_seq, val_mask, test_seq, test_mask


# setting seed for data_loaders for output reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
seed = 42 # any number 
set_deterministic(seed)

def data_loaders(train_seq, train_mask, train_y, val_seq, val_mask, val_y):
    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)

    # dataLoader for train set
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size, worker_init_fn=seed_worker)

    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_y)

    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_data)

    # dataLoader for validation set
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size, worker_init_fn=seed_worker)

    # wrap tensors
    test_data = TensorDataset(test_seq, test_mask, test_y)

    # sampler for sampling the data during training
    test_sampler = SequentialSampler(test_data)

    # dataLoader for validation set
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, worker_init_fn=seed_worker)

    return train_dataloader, val_dataloader, test_dataloader



# setting up the neural network for CodeBERT fine-tuning
class BERT_Arch(nn.Module):

    def __init__(self, bert):

        super(BERT_Arch, self).__init__()

        self.bert = auto_model

        # dropout layer
        self.dropout = nn.Dropout(0.2)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(2*768, 512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, output_layer)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=-1)

    # define the forward pass
    def forward(self, sent_id, mask):
        chunk_size = 512  # For chunking
        total_seq_length = sent_id.size(1)
        for start in range(0, total_seq_length, chunk_size):
            end = min(start + chunk_size, total_seq_length)
            chunk_sent_id = sent_id[:, start:end]
            chunk_mask = mask[:, start:end]
			
			# pass the inputs to the model
            cls_hs_current = self.bert(chunk_sent_id, attention_mask=chunk_mask)[1]
            if start == 0:
                cls_hs = cls_hs_current.clone()
            else:
                cls_hs = torch.cat([cls_hs, cls_hs_current], dim=1)

        fc1_output = self.fc1(cls_hs)
        relu_output = self.relu(fc1_output)
        dropout_output = self.dropout(relu_output)

        # output layer
        fc2_output = self.fc2(dropout_output)

        # apply softmax activation
        final_output = self.softmax(fc2_output)

        return final_output



# training the model
def train():

    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
            print('loss=',loss.item())

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds



def format_time(time=None, format=None, rebase=True):
    """See :meth:`I18n.format_time`."""
    return get_i18n().format_time(time, format, rebase)



# evaluating the model
def evaluate():

    print("\nEvaluating..")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:

            # Calculate elapsed time in minutes.
            # elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(
                step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

def get_evaluation_scores(tn, fp, fn, tp):
    print("get_score method is defined")
    if(tp == 0):
        accuracy = (tp+tn)/(tn+fp+fn+tp)
        Precision = 0
        Recall = 0
        F1 = 0
    else:
        accuracy = (tp+tn)/(tn+fp+fn+tp)
        Precision = tp/(tp+fp)
        Recall = tp/(tp+fn)
        F1 = 2*((Precision*Recall)/(Precision+Recall))
    return accuracy, F1, Precision, Recall

# define a batch size
batch_size = 8 

# give test data to the model in chunks to avoid Cuda out of memory error
def give_test_data_in_chunks(x_test):
    max_length = 1024
    n = len(x_test) / batch_size # because test_dataloader from bert_feature_extractor also uses batch_size
    preds_chunks = None
    for g, x_test_chunk in x_test.groupby(np.arange(len(x_test)) // n):
        tokens_test = tokenizer.batch_encode_plus(x_test_chunk.tolist(), max_length=max_length, pad_to_max_length=True, truncation=True)
        test_seq = torch.tensor(tokens_test['input_ids'])
        test_mask = torch.tensor(tokens_test['attention_mask'])
        preds_chunk = model(test_seq.to(device), test_mask.to(device))
        preds_chunk = preds_chunk.detach().cpu().numpy()
        preds_chunks = preds_chunk if preds_chunks is None else np.append(preds_chunks, preds_chunk, axis=0)
        preds = np.argmax(preds_chunks, axis=1)
    return preds

def bert_feature_extractor(test_dataloader):
    print("\nEvaluating..")
    # deactivate dropout layers
    model.eval()
    total_loss, total_accuracy = 0, 0
    # empty list to save the model predictions
    total_embeddings = []
    all_labels = []
    # iterate over batches
    for step, batch in enumerate(test_dataloader):
        # push the batch to gpu
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        with torch.no_grad():

            chunk_size = 512  # For chunking
            total_seq_length = sent_id.size(1)
            
            for start in range(0, total_seq_length, chunk_size):
                end = min(start + chunk_size, total_seq_length)
                chunk_sent_id = sent_id[:, start:end]
                chunk_mask = mask[:, start:end]
	    		
	    		# pass the inputs to the model
                cls_hs_current = model.bert(chunk_sent_id, attention_mask=chunk_mask)[1]
                if start == 0:
                    embeddings = cls_hs_current.clone()
                else:
                    embeddings = torch.cat([embeddings, cls_hs_current], dim=1)

            embeddings = embeddings.detach().cpu().numpy()
            total_embeddings.append(embeddings)
            label = labels.detach().cpu().numpy()
            all_labels.append(label)
    # reshape the predictions in form of (number of samples, no. of classes)
    total_embeddings = np.concatenate(total_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return total_embeddings, all_labels

#performing per project analysis

result = pd.DataFrame(columns = ['project_name','Accuracy','F1', 'Precision', 'Recall', 'TN', 'FP', 'FN', 'TP','avg_precision','avg_recall','avg_f1', 'total_test'])
execution_time_full = time.time()
print("Start time of  complete experiment", execution_time_full)
TN = FP = FN = TP = 0

x = 'full_code' # use the 'full_code' column to run Flakify using the full code instead of pre-processed code
if dataset_type == "IDoFT_6Cat":
    y = 'category' # use category instead of label_num
    output_layer=6
elif dataset_type == "IDoFT_2Cat":
    y = 'flaky' # for 2 category
    output_layer=2

project_index=0
total_weighted_avg_scores=[0, 0, 0]
total_support=0
weighted_avg_arrays_list=[]
category_dict={}
feature_extraction_time = 0
print(sorted(project_name))
for i in sorted(project_name):
    fit_time=0
    bert_flag=0
    project_index +=1
    
    project_Name=i
    
    train_dataset=  df.loc[(df['project'] != i)]
    test_dataset= df.loc[(df['project']== i)]

    train_x, valid_x, train_y, valid_y = train_test_split(train_dataset[x], train_dataset[y], 
                                                          random_state=49, 
                                                          test_size=0.2, 
                                                          stratify=train_dataset[y])
    test_x=test_dataset[x]

    test_y=test_dataset[y]

    tokens_train, tokens_val, tokens_test = tokenize_data(train_x, valid_x, test_x)

    # converting labels of train, validation and test into tensors

    Y_train = pd.DataFrame(train_y)
    y_val = pd.DataFrame(valid_y)
    y_test = pd.DataFrame(test_y)

    Y_train.columns = ['category']
    y_val.columns = ['category']
    y_test.columns = ['category']

    # converting labels of train, validation and test into tensors
    train_y = torch.tensor(Y_train['category'].values)
    val_y = torch.tensor(y_val['category'].values)
    test_y = torch.tensor(y_test['category'].values)

    train_seq, train_mask, val_seq, val_mask, test_seq, test_mask = text_to_tensors(tokens_train, tokens_val, tokens_test)

    # creating data_loaders for train and validation dataset
    train_dataloader, val_dataloader, test_dataloader = data_loaders(train_seq, train_mask, train_y, val_seq, val_mask, val_y)

     # compute the class weights
    class_weights = compute_class_weight('balanced', np.unique(Y_train.values), y=np.ravel(Y_train.values))
    # converting list of class weights to a tensor
    weights = torch.tensor(class_weights, dtype=torch.float)

    # push to GPU
    weights = weights.to(device)

    # define the loss function
    cross_entropy = nn.NLLLoss(weight=weights)

    # number of training epochs
    epochs = 20

    #print("Class Weights:", class_weights)

    model = BERT_Arch(auto_model)

    # push the model to GPU

    model = model.to(device)
    # define the optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    gc.collect()
    torch.cuda.empty_cache()
    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []
    print('Categories=',np.unique(y_test['category'].values))
    
    # load weights of best model
    print(model_weights_path+"_"+dataset_type+i+".pt")
    model.load_state_dict(torch.load(model_weights_path+"_"+dataset_type+i+".pt", map_location=device))
    #For quantization

    '''quantized_model = quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
        )
    #quantized_model.to(device)
    # Reassign the quantized_model to the global model variable
    torch.save(quantized_model.state_dict(), model_weights_path+str(i)+'_quantized_.pt')

    model = quantized_model'''
    #model.cpu()
    model.eval()  # Make sure model is in eval mode
     
    if ml_technique == "BERT":
        bert_flag=1
        start_time = time.time()
        with torch.no_grad():
            preds = give_test_data_in_chunks(test_x)
        inference_time=(time.time() - start_time)
        print('inference_time=',inference_time)

    elif ml_technique == "RF":
        print("RF....................")
        model_ml = RandomForestClassifier(n_estimators=1000, random_state=123456)#verbose=1, class_weight="balanced" )
    
    elif ml_technique == "SVC":
        model_ml = svm.SVC(C=1, kernel='linear', gamma='scale')

    elif ml_technique == "XGB":
        model_ml = XGBClassifier()
    
    elif ml_technique == "GBM":
        model_ml = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=1, random_state=0)
   
    elif ml_technique == "DT":
        model_ml = DecisionTreeClassifier(criterion='entropy', max_depth = None) 
    
    elif ml_technique == "MLP":
        model_ml = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=50) 

    elif ml_technique == "Ada":
        model_ml = AdaBoostClassifier(n_estimators=100, random_state=0) 
    
    elif ml_technique == "NB":
        model_ml = GaussianNB() 
    
    elif (ml_technique == 'KNN'):
        model_ml = KNeighborsClassifier(n_neighbors=7)
   
    elif (ml_technique == 'LR'):
        model_ml = LogisticRegression(random_state=0,solver='liblinear')

    elif (ml_technique == 'LDA'):
        model_ml = LinearDiscriminantAnalysis()

    if bert_flag == 0:

        x_train_bert, y_train_label = bert_feature_extractor(train_dataloader)

        start_time = time.time()
        trained_model = model_ml.fit(x_train_bert, y_train_label)
        fit_time+=(time.time() - start_time)
        print('fit_time=',fit_time)
        #pickle.dump(trained_model, open("filename_"+ml_technique+str(i), 'wb'))


        start_time = time.time()
        x_test_bert, y_test_label = bert_feature_extractor(test_dataloader)
        feature_extraction_time=(time.time() - start_time)

        start_time = time.time()
        preds = model_ml.predict(x_test_bert)
        inference_time=(time.time() - start_time)
        print('inference_time=',inference_time)

    print("test_y=",test_y.numpy())
    print("predicted=",preds)
    cr=classification_report(test_y, preds)
    print(cr)

    lines = cr.strip().split('\n')

    # parse the class names and metrics
    classes = []
    metrics = []
    for line in lines[2:-4]:  # skip the first 2 and last 3 lines
        t = line.strip().split()
        classes.append(t[0])
        key=t[0]
        values=[float(x) for x in t[1:]]

        with open("../Per_project_result/per_Category_Evaluation_"+dataset_type+".txt", "a") as file:
            file.write(i+":"+key+":" + str(values))
            file.write("\n")

        metrics.append(values)
        if key in category_dict:
            existing_values=category_dict[key]
            updated_values=[existing_values[k] + (values[k]*values[-1]) for k in range(len(values)-1)]
            updated_values.append(existing_values[-1] + values[-1]) #This is for adding support
            category_dict[key] = updated_values
        else:
            initial_val = [(values[i]*values[-1]) for i in range(len(values)-1)]
            initial_val.append(values[-1])
            category_dict[key] = initial_val

    
    print('metrics=',metrics)
    third_last_line = lines[-3].strip().split()

    accuracy = [float(x) for x in third_last_line[1:]]

    second_last_line = lines[-2].strip().split()
    macro_avg = [float(x) for x in second_last_line[2:]]

    # parse the overall scores
    last_line = lines[-1].strip().split()
    weighted_avg = [float(x) for x in last_line[2:]]
    # print the results
    print('Classes:', classes)
    #print('Metrics:', metrics)
    #print('Overall scores:', weighted_avg)

    total_weighted_avg_scores =  [ total_weighted_avg_scores[idx] + (weighted_avg[idx] * weighted_avg[-1]) for idx in range(3)] 
    #weighted_avg_arrays_list.append(weighted_avg)
    total_support +=weighted_avg[-1]

    with open("../Per_project_result/weighted_avg_per_project_test_"+dataset_type+".txt", "a") as file:
        file.write(i+",")
        file.write(str(weighted_avg))
        file.write(",feature_extraction_time="+str(feature_extraction_time))
        file.write(",inference_time="+str(inference_time))
        file.write(",fit_time="+str(fit_time))
        file.write("\n")

    del model
    torch.cuda.empty_cache()
    
avg_score=[round(i / (total_support), 4) for i in total_weighted_avg_scores] #
total_weighted_avg_scores.append((total_support))
avg_score.append(total_support)

with open("../Per_project_result/weighted_avg_per_project_test_"+dataset_type+".txt", "a") as file:
    file.write("Total_Weighted_score_for_all_project,")    
    file.write(str(total_weighted_avg_scores))
    file.write("\nWeighted_avg_score_for_all_project,")
    file.write(str(avg_score))
    file.write("\n")


with open("../Per_project_result/per_Category_Evaluation_"+dataset_type+".txt", "a") as file:
    file.write("WEIGHTED AVG==>")
    for key, value in category_dict.items():
        avg_score_per_category = [val/value[-1] for val in value[0:-1]]
        avg_score_per_category.append(value[-1])
        #file.write(key+",")
        file.write(key+":" + str(avg_score_per_category))
        file.write("\n")

