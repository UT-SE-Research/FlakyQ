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
    for category in range(4):
        y=np.where(test_y==category,1,0) #taking y only for the specific category
        print(y)
        #print(test_y)
        p=np.where(preds==category,1,0)
        print("P====:w,"+str(category))
        print(p)
        n_p=np.sum(y==1) #positive
        n_n=np.sum(y==0) #negative
        '''if (n_p == 0):
            recalls.append(-1000)
            precisions.append(-1000)
            f_scores.append(-1000)
            continue'''
        
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

#reading the parameters 

dataset_path = sys.argv[1]
model_weights_path = sys.argv[2]
results_file = sys.argv[3]
dataset_type = sys.argv[4] # IDoFT_2Cat/ IDoFT_6Cat

df = pd.read_csv(dataset_path)
df.head()
# get project names

project_name=['Chronicle-Wire', 'DataflowTemplates', 'Java-WebSocket', 'Mapper', 'admiral', 'adyen-java-api-library', 'biojava', 'esper', 'fastjson', 'hadoop', 'hbase', 'hive', 'innodb-java-reader', 'junit-quickcheck', 'mockserver', 'nacos', 'nifi', 'openhtmltopdf', 'ormlite-core', 'riptide', 'spring-boot', 'spring-data-r2dbc', 'spring-hateoas', 'typescript-generator', 'visualee', 'wildfly']
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
    # define a batch size
    batch_size = 8

    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)

    # dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, worker_init_fn=seed_worker)

    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_y)

    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_data)

    # dataLoader for validation set
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size, worker_init_fn=seed_worker)

    return train_dataloader, val_dataloader



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


        # pass the inputs to the model
        #cls_hs = self.bert(sent_id, attention_mask=mask)[1]

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



# give test data to the model in chunks to avoid Cuda out of memory error
def give_test_data_in_chunks(x_test):
    max_length = 1024
    n = len(x_test) / 50 
    preds_chunks = None
    for g, x_test_chunk in x_test.groupby(np.arange(len(x_test)) // n):
        tokens_test = tokenizer.batch_encode_plus(
            x_test_chunk.tolist(), max_length=max_length, pad_to_max_length=True, truncation=True)
        test_seq = torch.tensor(tokens_test['input_ids'])
        test_mask = torch.tensor(tokens_test['attention_mask'])
        preds_chunk = model(test_seq.to(device), test_mask.to(device))
        preds_chunk = preds_chunk.detach().cpu().numpy()
        preds_chunks = preds_chunk if preds_chunks is None else np.append(preds_chunks, preds_chunk, axis=0)
        preds = np.argmax(preds_chunks, axis=1)

    return preds


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

#x='full_code'
#y='category'

project_index=0
print(sorted(project_name))

for i in sorted(project_name):

    project_index +=1

    #if project_index < 98:
    #    continue
    print(str(project_index)+' testing on project: ', i)
    project_Name=i
    
    train_dataset=  df.loc[(df['project'] != i)]
    test_dataset= df.loc[(df['project']== i)]
    #train_dataset[['project','flaky']].to_csv('Train_dataset.csv', index=False)
    #test_dataset[['project','flaky']].to_csv('Test_dataset.csv', index=False)
    train_x, valid_x, train_y, valid_y = train_test_split(train_dataset[x], train_dataset[y], random_state=49, test_size=0.2, stratify=train_dataset[y])
    test_x=test_dataset[x]
    test_y=test_dataset[y]

    print(test_y)
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
    train_dataloader, val_dataloader = data_loaders(train_seq, train_mask, train_y, val_seq, val_mask, val_y)

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

    print("Class Weights:", class_weights)

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
    print(np.unique(y_test['category'].values))
    # for each epoch
    for epoch in range(epochs):

        #break
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        # train the model
        train_loss, _ = train()

        # evaluate the model
        valid_loss, _ = evaluate()

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_weights_path+"_"+dataset_type+i+".pt")

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

    # load weights of best model
    model.load_state_dict(torch.load(model_weights_path+"_"+dataset_type+i+".pt"))
    print("The training process for each project is completed in : (%s) seconds. " % round((time.time() - execution_time_full), 5))

    
    with torch.no_grad():
        preds = give_test_data_in_chunks(test_x)


    cr=classification_report(test_y, preds)
    print(type(cr))
    #with open("Flakicat_classification_report_per_project.txt", "a") as file:
    with open("classification_report_per_project.txt", "a") as file:
        file.write("Epoch="+str(epoch)+",Project_name="+i+"\n")
        file.write(cr)
        file.write("\n")

    cm = confusion_matrix(test_y, preds)
    #print(cm)
	
    with open("confusion_matrix_per_project.txt", "a") as file:
        file.write("Epoch="+str(epoch)+",Project_name="+i+"\n")
        file.write(np.array2string(cm))
        file.write("\n")
        
    TP, TN, FP, FN, Precisions, Recalls, F1, accuracy = compute_metrics(test_y, preds)
    
    avg_precision=calculate_filtered_avg(Precisions, th=-1000)
    avg_recall=calculate_filtered_avg(Recalls, th=-1000)
    avg_f1= calculate_filtered_avg(F1, th=-1000)

    del model
    torch.cuda.empty_cache()
    

    result = result.append(pd.Series([project_Name,accuracy, F1, Precisions, Recalls, TN, FP, FN, TP, avg_precision,avg_recall,avg_f1, len(test_y)], index=result.columns), ignore_index=True)

    result.to_csv(results_file,  index=False)


