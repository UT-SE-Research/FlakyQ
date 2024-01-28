import os
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
import scipy as sp
import pickle
import sklearn.metrics as metrics
import torch.nn.utils.prune as prune
import torch.quantization
import torch.quantization as quantization
from torch.quantization import fuse_modules

def parse_cr(cr, ml_technique, fold, feature_extraction_time, prediction_time, fit_time):
    global total_weighted_avg_scores
    global total_support
    global weighted_avg_arrays_list
    global category_dict

    lines = cr.strip().split('\n')

    # parse the class names and metrics

    classes = []
    metrics = []
    for line in lines[2:-4]:  # skip the first 2 and last 3 lines
        t = line.strip().split()
        classes.append(t[0])
        key=t[0]
        values=[float(x) for x in t[1:]]

        with open("../flaky-test-categorization/per_Category_Evaluation_"+technique+".txt", "a") as file:
            file.write(fold+":"+key+":" + str(values))
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

    #print(category_dict[key]/category_dict[key][-1]) 
    #print("*******************************************")
    #print('metrics=',metrics)
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

    with open("../flaky-test-categorization/weighted_avg_for_cv_"+technique+"_without_quantization.txt", "a") as file: # Once I get the result, need to divide by 10
        file.write(fold+",")
        file.write(str(weighted_avg))
        file.write(","+feature_extraction_time+","+prediction_time)
        file.write(",fit_time="+fit_time)
        file.write("\n")

def fmt_gpu_mem_info(gpu_id=0, brief=True) -> str:
    import torch.cuda.memory

    if torch.cuda.is_available():
        report = ""
        t = torch.cuda.get_device_properties(gpu_id).total_memory
        c = torch.cuda.memory.memory_reserved(gpu_id)
        a = torch.cuda.memory_allocated(gpu_id)
        f = t - a

        report += f"[Allocated {a} | Free {f} | Cached {c} | Total {t}]\n"
        if not brief:
            report += torch.cuda.memory_summary(device=gpu_id, abbreviated=True)
        return report
    else:
        return f"CUDA not available, using CPU"

# setting the seed for reproducibility, same seed is set to ensure the reproducibility of the result
def set_deterministic(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True 

# specify GPU
#device = torch.device("cuda")
device = torch.device("cpu")

# read the parameters 

dataset_path = sys.argv[1]
print(dataset_path)
model_weights_path = sys.argv[2]
#results_file = sys.argv[3]
technique = sys.argv[4]
ml_technique=technique.split("-")[0]
which_dataset=technique.split("-")[1]
#print(technique)
data_name = sys.argv[5]
where_data_comes=data_name.split("-")[0]
dataset_category=technique.split("-")[1]

df = pd.read_csv(dataset_path)
input_data = df['full_code'] # use the 'full_code' column to run Flakify using the full code instead of pre-processed code
#print(len(input_data))
print('****************',which_dataset)
if which_dataset == "Flakicat":
    print("It is Flakicat")
    out_layer = 5 # if Flakicat dataset
    target_data = df['category']

elif which_dataset == "IDoFT_6Cat" or dataset_category == "IDoFT_6CatStructurePrune" or dataset_category == "IDoFT_6CatUnStructurePrune":
    out_layer = 6 # if Flakicat dataset
    target_data = df['category']

elif dataset_category == "IDoFT_2Cat":
    print('From elif****************',which_dataset)
    out_layer=2
    target_data = df['flaky'] # For multi-class

df.head()

# balance dataset,converts a 1D Pandas DataFrame into a 2D NumPy array
def sampling(X_train, y_train, X_valid, y_valid):
    
    oversampling = RandomOverSampler(
        sampling_strategy='minority', random_state=49) #RandomOverSampler generates synthetic samples for the minority class, making the class distribution more balanced
    x_train = X_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    x_val = X_valid.reshape(-1, 1)
    y_val = y_valid.reshape(-1, 1)
    
    x_train, y_train = oversampling.fit_resample(x_train, y_train)
    x_val, y_val = oversampling.fit_resample(x_val, y_val)
    x_train = pd.Series(x_train.ravel())
    y_train = pd.Series(y_train.ravel())
    x_val = pd.Series(x_val.ravel())
    y_val = pd.Series(y_val.ravel())

    return x_train, y_train, x_val, y_val


# define CodeBERT model
model_name = "microsoft/codebert-base"
model_config = AutoConfig.from_pretrained(model_name, return_dict=False, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
auto_model = AutoModel.from_pretrained(model_name, config=model_config)

# convert code into tokens and then vector representation
def tokenize_data(train_text, val_text, test_text):
    #max_length = 1024
    max_length = 512
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

    #print(tokens_train)

    return tokens_train, tokens_val, tokens_test

# convert vector representation to tensors
def text_to_tensors(tokens_train, tokens_val, tokens_test):
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])

    return train_seq, train_mask, val_seq, val_mask, test_seq, test_mask

# sett seed for data_loaders for output reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
seed = 42 # any number 
set_deterministic(seed)

def data_loaders(train_seq, train_mask, train_y, val_seq, val_mask, val_y):
    # define a batch size
    batch_size = 1
    print('From data_loader')
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

# set up the neural network for CodeBERT fine-tuning
class BERT_Arch(nn.Module):

    def __init__(self, bert):

        super(BERT_Arch, self).__init__()

        self.bert = auto_model

        # dropout layer
        self.dropout = nn.Dropout(0.2)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        #self.fc1 = nn.Linear(2*768, 512) # For 1024 tokens
        self.fc1 = nn.Linear(768, 512) #for 512 tokens
        #prune.l1_unstructured(self.fc1, name="weight", amount=0.2)  # Example pruning

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, out_layer)
        #prune.l1_unstructured(self.fc2, name="weight", amount=0.2)  # Example pruning

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=-1)

    # define the forward pass
    def forward(self, sent_id, mask):

        chunk_size = 512  # For chunking
        #overlap_size = 256
        total_seq_length = sent_id.size(1)
        #print('***************=',total_seq_length)
        # Split the sequence into chunks of size chunk_size
        #cls_hs_list = []
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
        
        #cls_hs = self.bert(sent_id, attention_mask=mask)[1]
        #print(cls_hs.shape)

        fc1_output = self.fc1(cls_hs)

        relu_output = self.relu(fc1_output)

        dropout_output = self.dropout(relu_output)

        # output layer
        fc2_output = self.fc2(dropout_output)

        # apply softmax activation
        final_output = self.softmax(fc2_output)

        return final_output

# train the model, Fine-tune
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


def bert_feature_extractor(my_dataloader):
    print("\nEvaluating..")
    # deactivate dropout layers
    model.eval()
    total_loss, total_accuracy = 0, 0
    # empty list to save the model predictions
    total_embeddings = []
    all_labels = []
    # iterate over batches
    for step, batch in enumerate(my_dataloader):
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

            # model predictions
            #embeddings = model.bert(sent_id, mask)[1]  # calling to the model
            embeddings = embeddings.detach().cpu().numpy()
            total_embeddings.append(embeddings)
            label = labels.detach().cpu().numpy()
            all_labels.append(label)
    # reshape the predictions in form of (number of samples, no. of classes)
    total_embeddings = np.concatenate(total_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return total_embeddings, all_labels

def format_time(time=None, format=None, rebase=True):
    """See :meth:`I18n.format_time`."""
    return get_i18n().format_time(time, format, rebase)

# evaluate the model
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


'''def give_test_data_in_chunks(x_test_nparray): #BERT
    print(x_test_nparray.shape)
    max_length = 1024
    x_test = pd.DataFrame(x_test_nparray) 
    n = len(x_test) / 50 
    preds_chunks = None
    for g, x_test_chunk in x_test.groupby(np.arange(len(x_test)) // n):
        tokens_test = tokenizer.batch_encode_plus(
            x_test_chunk.squeeze().tolist(), max_length=max_length, pad_to_max_length=True, truncation=True)
        test_seq = torch.tensor(tokens_test['input_ids'])
        test_mask = torch.tensor(tokens_test['attention_mask'])
        preds_chunk = model(test_seq.to(device), test_mask.to(device))
        preds_chunk = preds_chunk.detach().cpu().numpy()
        preds_chunks = preds_chunk if preds_chunks is None else np.append(
            preds_chunks, preds_chunk, axis=0)
        preds = np.argmax(preds_chunks, axis=1)

    return preds'''

# give test data to the model in chunks to avoid Cuda out of memory error
def give_test_data_in_chunks(x_test_nparray): #BERT
    #max_length = 1024
    max_length = 512
    x_test = pd.DataFrame(x_test_nparray)
    n = len(x_test) / 50
    preds_chunks = None
    for g, x_test_chunk in x_test.groupby(np.arange(len(x_test)) // n):
        # Ensure x_test_chunk is a DataFrame
        if isinstance(x_test_chunk, pd.Series):
            x_test_chunk = x_test_chunk.to_frame().T

        # Convert DataFrame to list of strings
        test_data = x_test_chunk.iloc[:, 0].tolist() if len(x_test_chunk) > 1 else [x_test_chunk.iloc[0, 0]]

        tokens_test = tokenizer.batch_encode_plus(
            test_data, max_length=max_length, pad_to_max_length=True, truncation=True)
        test_seq = torch.tensor(tokens_test['input_ids'])
        test_mask = torch.tensor(tokens_test['attention_mask'])
        preds_chunk = model(test_seq.to(device), test_mask.to(device))
        preds_chunk = preds_chunk.detach().cpu().numpy()
        preds_chunks = preds_chunk if preds_chunks is None else np.append(preds_chunks, preds_chunk, axis=0)
    preds = np.argmax(preds_chunks, axis=1)

    return preds



execution_time = time.time()
print("Start time of the experiment", execution_time)
no_split=10 # For Flakicat (10 folds)
skf = StratifiedKFold(n_splits=no_split,shuffle=True)
TN = FP = FN = TP = 0
fold_number = 0

#for train_index, test_index in skf.split(input_data, target_data):
total_weighted_avg_scores=[0, 0, 0]
total_support=0
weighted_avg_arrays_list=[]
category_dict={}
feature_extraction_time=0
for fold in range(no_split):
    fit_time=0
    bert_flag=0
    total_execution_time = 0
    #total_execution_time_for_feature_extraction = 0
    print(" NOW IN FOLD NUMBER", fold_number)
    print('data_name=',data_name)
    #for q in range(len(test_index)):
        #print(test_index[q])
    #X_train, X_test = input_data.iloc[list(train_index)], input_data.iloc[list(test_index)]
    #y_train, y_test = target_data.iloc[list(train_index)], target_data.iloc[list(test_index)]
    #print(data_name+'/data_splits/X_test_fold'+str(fold)+'.npy')
    X_test=np.load(data_name+'/data_splits/X_test_fold'+str(fold)+'.npy',allow_pickle=True)
    y_test=np.load(data_name+'/data_splits/y_test_fold'+str(fold)+'.npy',allow_pickle=True)
    
#    print(')
    X_train=np.load(data_name+'/data_splits/X_train_fold'+str(fold)+'.npy',allow_pickle=True)
    y_train=np.load(data_name+'/data_splits/y_train_fold'+str(fold)+'.npy',allow_pickle=True)

    X_valid=np.load(data_name+'/data_splits/X_valid_fold'+str(fold)+'.npy',allow_pickle=True)
    y_valid=np.load(data_name+'/data_splits/y_valid_fold'+str(fold)+'.npy',allow_pickle=True)

    #X_train, X_valid, y_train, y_valid=train_test_split(X_train, y_train, random_state=49, test_size=0.2, stratify=y_train)
    #X_train, y_train, X_valid, y_valid = sampling(X_train, y_train, X_valid, y_valid)

    tokens_train, tokens_val, tokens_test = tokenize_data(X_train, X_valid, X_test)
    Y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_valid)
    y_test = pd.DataFrame(y_test)

    Y_train.columns = ['which_tests']
    y_val.columns = ['which_tests']
    y_test.columns = ['which_tests']

    # convert labels of train, validation and test into tensors
    train_y = torch.tensor(Y_train['which_tests'].values)
    val_y = torch.tensor(y_val['which_tests'].values)
    test_y = torch.tensor(y_test['which_tests'].values)

    train_seq, train_mask, val_seq, val_mask, test_seq, test_mask = text_to_tensors(tokens_train, tokens_val, tokens_test)

    # create data_loaders for train and validation dataset
    train_dataloader, val_dataloader, test_dataloader = data_loaders(train_seq, train_mask, train_y, val_seq, val_mask, val_y)

    # compute the class weights
    class_weights = compute_class_weight('balanced', np.unique(Y_train.values), y=np.ravel(Y_train.values))
    # convert list of class weights to a tensor
    weights = torch.tensor(class_weights, dtype=torch.float)

    # push to GPU
    weights = weights.to(device)

    # define the loss function
    cross_entropy = nn.NLLLoss(weight=weights)

    # number of training epochs
    epochs = 20

    print("Class Weights:", class_weights)

    '''if "Prune" in dataset_category: 
        print('FROM PRUNE')
        model_pruned = BERT_Arch_Pruned(auto_model)
        structured_prune_model(model_pruned)
    else:'''
    model = BERT_Arch(auto_model)

    
    # push the model to GPU
    model = model.to(device)

    # define the optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)
    #optimizer = AdamW(model.parameters(), lr=1e-5)

    gc.collect()
    torch.cuda.empty_cache()
    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    # load weights of best model
    #print(model_weights_path+str(fold_number)+'_pruned.pt')

    '''if "Prune" in dataset_category: 
        model.load_state_dict(torch.load(model_weights_path+str(fold_number)+'_structured_pruned.pt'))
    else:'''
    print(model_weights_path+str(fold_number)+'.pt')
    model.load_state_dict(torch.load(model_weights_path+str(fold_number)+'_original_flakify.pt'))
    #fuse_modules(model, ['fc1', 'relu'], inplace=True)
    quantized_model = quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    #quantized_model.to(device)
    # Reassign the quantized_model to the global model variable
    torch.save(quantized_model.state_dict(), model_weights_path+str(fold_number)+'_quantized_.pt')
    model = quantized_model
    #model.cpu()
    model.eval()  # Make sure model is in eval mode
    #model.to(device)

    model.to(device)

    model.eval()  # Make sure model is in eval mode
    required_time = 0
    if ml_technique == "BERT":
       print ("I AM RUNNING FROM TESTING_BERT and ml_technique is BERT**************")
       bert_flag=1
       #start_time = time.time()
       with torch.no_grad():
           preds = give_test_data_in_chunks(X_test)

       #required_time = time.time() - start_time
       #print('required_time=',required_time)
       #cr=classification_report(test_y, preds)
       #parse_cr(cr, technique, str(fold), str(feature_extraction_time), str(total_execution_time), str(fit_time))
    
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

    #print('y shape=',y_train_label.shape)
    #x_train_bert_np=np.array(x_train_bert)
    #y_train_label_np=np.array(y_train_label)
    #model.cpu()

    start_time = time.time()
    x_test_bert, y_test_label = bert_feature_extractor(test_dataloader)
    feature_extraction_time+=(time.time() - start_time)
    print('feature_extraction_time=',feature_extraction_time)

    if bert_flag == 0:
        print('***********loading done************')
        x_train_bert, y_train_label = bert_feature_extractor(train_dataloader)
        print(x_train_bert.shape)
        start_time = time.time()
        trained_model = model_ml.fit(x_train_bert, y_train_label)
        fit_time+=(time.time() - start_time)
        print('fit_time=',fit_time)

        #torch.save(trained_model.state_dict(), model_weights_path+str(fold_number)+ml_technique+'.pt')
        pickle.dump(trained_model, open("filename_"+ml_technique+str(fold_number), 'wb'))

        start_time = time.time()
        preds = model_ml.predict(x_test_bert)
        total_execution_time+=(time.time() - start_time)
        print('total_inference_time=',total_execution_time)

    cr=classification_report(y_test_label, preds)
    parse_cr(cr, technique, str(fold), str(feature_extraction_time), str(total_execution_time), str(fit_time))

    #print(type(cr))
    with open(where_data_comes+"-result/"+technique+"_classification_report.txt", "a") as file:
        file.write(technique+",Fold="+str(fold_number)+"\n")
        file.write(cr)
        file.write("\n")
        #file.write(","+str(total_execution_time))

    cm = confusion_matrix(y_test_label, preds)
    #print(cm)
	
    with open(where_data_comes+"-result/"+technique+"_confusion_matrix_val.txt", "a") as file:
        file.write(technique+",Fold="+str(fold_number)+"\n")
        file.write(np.array2string(cm))
        file.write("\n")
    
    # Create a heatmap plot of the confusion matrix
    #fig, ax = plt.subplots(figsize=(10, 8))
    #sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
    
    # Set the plot title and labels
    #ax.set_xlabel("Predicted labels")
    #ax.set_ylabel("True labels")
    #ax.set_title("Confusion Matrix")
    
    # Set the axis tick labels
    #class_names = [0, 1, 2, 3, 4, 5] # replace with your own class names
    #ax.xaxis.set_ticklabels(class_names)
    #ax.yaxis.set_ticklabels(class_names)
    
    #plt.show()
    #plt.savefig('Confusion-Matrix/confusion_matrix_val_fold_'+str(fold_number)+'.png')
    
    del model
    print("delete model")
    torch.cuda.empty_cache()

    fold_number = fold_number+1

for cls in range(out_layer):
    cls=str(cls)
    print(cls)
    avg_category_dict =  [(category_dict[cls][idx] /category_dict[cls][-1]) for idx in range(3)] 
    print(avg_category_dict)
    with open("../flaky-test-categorization/per_Category_Evaluation_"+technique+"_without_quantization.txt", "a") as file:
    #with open("../flaky-test-categorization/per_Category_Evaluation_"+technique+"pruned.txt", "a") as file:
        file.write(cls+":" + str(avg_category_dict))
        file.write("\n")


print("ml_technique="+ml_technique)
print("need to divide by 10,fit_time="+str(fit_time))
print("need to divide by 10, feature_extraction_time="+str(feature_extraction_time))
#print("The processed is completed in : (%s) seconds. " % round((time.time() - execution_time), 5))






