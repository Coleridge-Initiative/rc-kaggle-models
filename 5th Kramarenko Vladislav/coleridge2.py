import warnings,os,re,glob,json,sys,torch,random
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from collections import Counter

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch.nn as nn

from tqdm import tqdm

from transformers import BertTokenizerFast, AutoModelForTokenClassification


def read_append_return(filename, train_files_path='train', output='text'):
    """
    Function to read json file and then return the text data from them and append to the dataframe
    """
    json_path = os.path.join(train_files_path, (filename+'.json'))
    headings = []
    contents = []
    combined = []
    with open(json_path, 'r') as f:
        json_decode = json.load(f)
        for data in json_decode:
            headings.append(data.get('section_title'))
            contents.append(data.get('text'))
            combined.append(data.get('section_title'))
            combined.append(data.get('text'))
    
    all_headings = ' '.join(headings)
    all_contents = ' '.join(contents)
    all_data = '. '.join(combined)
    
    if output == 'text':
        return all_contents
    elif output == 'head':
        return all_headings
    else:
        return all_data


def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()
    
    
def text_cleaning(text):
    '''
    Converts all text to lower case, Removes special charecters, emojis and multiple spaces
    text - Sentence that needs to be cleaned
    '''
    text = re.sub('[^A-Za-z0-9\.\:\,\!\?\;\&]+', ' ', str(text)).strip()
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub('\s', ' ', text)
    return text.strip()
    

class VTBDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, size_s, labels = None):
        self.encodings = encodings
        self.labels = labels
        self.size_s = size_s
    def __getitem__(self, idx):
        self.item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            self.item['labels'] = torch.tensor(self.labels[idx])
        return self.item

    def __len__(self):
        return self.size_s


def encode_tags(labels, encodings):
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)
        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels[:sum((arr_offset[:,0] == 0) & (arr_offset[:,1] != 0))]
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels
  
  
def generate():  
    model.eval()
    sub = {}
    bad_w = ['consortium','organization','bureau','development','center','table','department','university','bank','class','user'
         'appendix','supplementary','supplement','major','association','journal','commission','associates','board','agency',
        'administration','federation','ministry','form','score','management','accounts','account','feasibility']
    good_word = ['resource','report','research','survey','agriculture','service',"study","database","program","data","dataset","assessment",'monitoring','surveys','initiative','system','student',
        'observation','census','directory','reports','statistics','codes','student','students','baccalaureate','sample','project','initiatives']
    for index, row in sample_sub.iterrows():#
        ans = []
        sample_text = text_cleaning(row['text'])
        sample_text = sample_text.split(". ")
        
        sent = ''
        new_s = []
        for i,sentens in enumerate(sample_text):
            if not sent:
                
                if len(sentens)>200 and len(sentens)<400:
                    new_s.append(sentens)
                elif len(sentens)>=400:
                    new_s.extend(re.findall(r'(?: |^).{0,150}[A-Z][a-z]{2,20} (?:(?:[A-Z][a-z]{2,20}|of|up|to|and|the|in|on|COVID-19|s|for|[0-9]{4}})[- \.,]){0,10}(?:[A-Z][a-z]{2,20})(?: data| survey| sample| study| [0-9]{2,4})*.{0,150}(?:[\. ]|$)', sentens))
                    new_s.extend(re.findall(r'(?: |^).{0,200}(?: [Dd]ata| [Rr]egistry|[Gg]enome [Ss]equence| [Mm]odel| [Ss]tudy| [Ss]urvey).{0,200}(?:[\. ]|$)', sentens))
                    new_s.extend(re.findall(r'(?: |^).{0,200}[A-Z]{4,10}.{0,200}(?:[\. ]|$)', sentens))                
                else:
                    sent = sentens
            else:
                if len(sent + sentens) >= 400:
                    new_s.append(sent)
                    sent = ''
                    if len(sentens)>200 and len(sentens)<400:
                        new_s.append(sentens)
                    elif len(sentens)>=400:
                        new_s.extend(re.findall(r'(?: |^).{0,150}[A-Z][a-z]{2,20} (?:(?:[A-Z][a-z]{2,20}|of|up|to|and|the|in|on|COVID-19|s|for|[0-9]{4}})[- \.,]){0,10}(?:[A-Z][a-z]{2,20})(?: data| survey| sample| study| [0-9]{2,4})*.{0,150}(?:[\. ]|$)', sentens))
                        new_s.extend(re.findall(r'(?: |^).{0,200}(?: [Dd]ata| [Rr]egistry|[Gg]enome [Ss]equence| [Mm]odel| [Ss]tudy| [Ss]urvey).{0,200}(?:[\. ]|$)', sentens))
                        new_s.extend(re.findall(r'(?: |^).{0,200}[A-Z]{4,10}.{0,200}(?:[\. ]|$)', sentens))    
                    else:
                        sent = sentens
                else:
                    sent = sent +'. ' + sentens
        if sent:
            new_s.append(sent)
        
        new_s_2 = []
        for s in new_s:
            
            a = re.findall(r'(?:(?:[A-Z][a-z]{2,20}|of|in|COVID-19|s|for|and) ){3,6}', s)
            
            a.extend(re.findall(r'(?: [Dd]ata| [Rr]egistry|[Gg]enome [Ss]equence| [Mm]odel| [Ss]tudy| [Ss]urvey)', s))
            a.extend(re.findall(r'[A-Z]{4,10}', s))
            if a:
                new_s_2.append(s)
        if new_s_2:
            t_x = [s.split() for s in new_s_2]
            valid_y = [[1]*len(x) for x in t_x]
            
            val_encodings = tokenizer(t_x, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True,max_length = 256)    
            val_labels = encode_tags(valid_y, val_encodings)    
            val_encodings.pop("offset_mapping")
            
            valid_dataset = VTBDataset(val_encodings,len(t_x),val_labels)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size_test, shuffle=False)
            
            valid = pd.DataFrame()
            valid['val_labels'] = val_encodings['input_ids']
            val_labels = len(val_encodings['input_ids'])
            len_val = len(val_encodings['input_ids'][0])

            valid_preds1 = np.zeros((val_labels,len_val), dtype = np.float32)
            valid_preds2 = np.zeros((val_labels,len_val), dtype = np.float32)

            avg_accuracy = 0.
            with torch.no_grad():
                for i,(batch)  in enumerate(valid_loader):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels']
                    outputs = model(input_ids, attention_mask=attention_mask, labels=None)

                    logits1 = outputs[0][:,:,1].detach()
                    logits1[labels<0] = -10
                    logits2 = outputs[0][:,:,2].detach()
                    logits2[labels<0] = -10                   
                    valid_preds2[i*batch_size_test:(i+1)*batch_size_test,:]=logits2.cpu().numpy()  
                    valid_preds1[i*batch_size_test:(i+1)*batch_size_test,:]=logits1.cpu().numpy()                             

        
        ans = []
        for index, row_1 in valid.iterrows():
        
            preds1, = np.where(valid_preds1[index]>1)
            preds2, = np.where(valid_preds2[index]>0)
            
            preds2.sort()
            max_a = 0
            g_all = []
            for min_a in preds1:
                if max_a > min_a:
                    continue
                g = ''
                max_a = 0
                for min_b in preds2:
                    if min_b>min_a:
                        max_a = min_b
                        break
                if max_a == 0 and valid_preds1[index][min_a]>2:
                    max_a = min_a  
                if max_a-min_a > 10:
                    continue
                if max_a>=min_a and min_a>0:
                    k = 0
                      
                    b = np.array(row_1["val_labels"])

                    s = tokenizer.convert_ids_to_tokens(b[min_a:])
                    for j,w in enumerate(s):
                        
                        if j<=max_a - min_a or "##" in w  :
                            g += w + ' '
                        else:    
                            break
                g = g.replace(" ##","").strip()
                
                if g and sum(map(str.isupper,g))/len(g.split())>0.5:
                    g = clean_text(g)
                    it_bad = False
                    for w2 in g.split():
                        if w2 in bad_w: 
                            it_bad = True
                    if not it_bad:
                        for w in g.split():
                            if w in good_word or len(g.split())==1:
                                g_all.append(g)
                                break            
        
            ans.extend(g_all)

        if not row['Id'] in sub:
            sub[row['Id']] = ''
        for s in ans:
            if len(sub[row['Id']]) and not s in sub[row['Id']]:
                it_bad = False
                for p in sub[row['Id']].split("|"):
                    if p in s:
                        it_bad = True
                if not it_bad:              
                    sub[row['Id']] = sub[row['Id']] + '|'+ s
            elif not len(sub[row['Id']]):
                sub[row['Id']] = s 
    print(sub)




def train(train_df): 
    bads = 0
    t_x = []
    t_y = []
    f_x = []
    f_y = []
    t_l = 0
    f_l = 0
    valid_x = []
    valid_y = []
    valid_l = []    
    for index, row in tqdm(train_df.iterrows(),position = 0):
        sample_text = text_cleaning(row['text'])
        sample_text = sample_text.split(". ")
        
        dataset_label = clean_text(row['dataset_label'])
       
        sent = ''
        new_s = []
        for i,sentens in enumerate(sample_text):
            if not sent:
            
                if len(sentens)>200 and len(sentens)<400:
                    new_s.append(sentens)
                elif len(sentens)>=400:
                    new_s.extend(re.findall(r'(?: |^).{0,150}(?:(?:[A-Z][a-z]{2,20}(?:\'s)*|of|in|COVID-19|for|and)[- \.,]){3,6}(?:\([A-Z]+\) )*.{0,150}(?:[\. ]|$)', sentens))
                    new_s.extend(re.findall(r'(?: |^).{0,200}(?: [Dd]ata| [Rr]egistry|[Gg]enome [Ss]equence| [Mm]odel| [Ss]tudy| [Ss]urvey).{0,200}(?:[\. ]|$)', sentens))
                    new_s.extend(re.findall(r'(?: |^).{0,200}[A-Z]{4,10}.{0,200}(?:[\. ]|$)', sentens))               
                else:
                    sent = sentens
            else:
                if len(sent + sentens) >= 400:
                    new_s.append(sent)
                    sent = ''
                    if len(sentens)>200 and len(sentens)<400:
                        new_s.append(sentens)
                    elif len(sentens)>=400:
                        new_s.extend(re.findall(r'(?: |^).{0,150}(?:(?:[A-Z][a-z]{2,20}(?:\'s)*|of|in|COVID-19|for|and)[- \.,]){3,6}(?:\([A-Z]+\) )*.{0,150}(?:[\. ]|$)', sentens))
                        new_s.extend(re.findall(r'(?: |^).{0,200}(?: [Dd]ata| [Rr]egistry|[Gg]enome [Ss]equence| [Mm]odel| [Ss]tudy| [Ss]urvey).{0,200}(?:[\. ]|$)', sentens))
                        new_s.extend(re.findall(r'(?: |^).{0,200}[A-Z]{4,10}.{0,200}(?:[\. ]|$)', sentens))              
                    else:
                        sent = sentens
                else:
                    sent = sent +'. ' + sentens
        if sent:
            new_s.append(sent)
        
        new_s_2 = []
        for s in new_s:
            
            a = re.findall(r'(?:(?:[A-Z][a-z]{2,20}|of|in|COVID-19|s|for|and) ){3,6}', s)
            
            a.extend(re.findall(r'(?: [Dd]ata| [Rr]egistry|[Gg]enome [Ss]equence| [Mm]odel| [Ss]tudy| [Ss]urvey)', s))
            a.extend(re.findall(r'[A-Z]{4,10}', s))
            if a:
                new_s_2.append(s)

        for s in new_s_2:

            good_t = ''
            y = []

            s1 = s.lower()
            y = [0]*len(s.split())
            for t in existing_labels:

                if t in s1:

                    if t:
                        s = re.sub('(?<='+t+')',' ',s,flags=re.IGNORECASE)
                        s = re.sub('(?='+t+')',' ',s,flags=re.IGNORECASE)
                    s = re.sub('\s',' ',s)
                    s1 = s.lower()           
                    y = [0]*len(s.split())

                    if len(t)<6:
                        result = re.split('(?<=[\s^])'+t+'(?=[\s$])', s1)#
                    else:
                        result = re.split(t, s1)

                    if len(result)>1:
                        good_t=t
                        m = 0

                        for r in result[:-1]:
                            len_t = len(t.split())
                            m = m+len(r.split())

                            y[m+len_t-1] = 2
                            y[m] = 1

                            m = m+len_t

            if row['Id'][0]=='0':
                if max(y)>0:
                    t_l += 1
                    valid_l.append(good_t)
                    valid_x.append(s.split())
                    valid_y.append(y)
                elif t_l>f_l:
                    f_l += 1
                    valid_l.append('')
                    valid_x.append(s.split())
                    valid_y.append(y)
            else:
                if max(y)>0:
                    t_x.append(s.split())
                    t_y.append(y)
                else:
                    f_x.append(s.split())
                    f_y.append(y)            


    print(valid_y[:10])
    print(valid_l[:10])
    print(len(t_x),bads,len(f_x))

    val_encodings = tokenizer(valid_x, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True,max_length = 200)
    val_labels = encode_tags(valid_y, val_encodings)    
    val_encodings.pop("offset_mapping")

    valid_dataset = VTBDataset(val_encodings,len(val_labels), val_labels)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size_test, shuffle=False)
    valid = pd.DataFrame()
    valid['val_labels'] = val_encodings['input_ids']
    valid['label'] = valid_l
    valid['valid_x'] = [" ".join(x) for x in valid_x]
    len_val = len(val_encodings['input_ids'][0])

    N_t = len(t_x)//10
    N_f = len(f_x)
    tq = tqdm(range(EPOCHS))

    for epoch in tq:
        model.float()
        model.train()
        avg_loss = 0.
        avg_accuracy = 0.
        lossf=None
        t_x_all = t_x[:]
        t_y_all = t_y[:] 
        start_f = random.randint(0,N_f//N_t)
        t_x_all.extend(f_x[start_f::N_f//N_t])
        t_y_all.extend(f_y[start_f::N_f//N_t])
        print(len(t_x_all))
        train_encodings = tokenizer(t_x_all, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True,max_length = 200)    
        train_labels = encode_tags(t_y_all, train_encodings)    
        train_encodings.pop("offset_mapping")
        train_dataset = VTBDataset(train_encodings,len(train_labels), train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)    
        tk0 = tqdm(enumerate(train_loader),total=len(train_loader),leave=False,position = 0)
        optimizer.zero_grad()
        
        for i,(batch) in tk0:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()                             
            optimizer.zero_grad()            
            logits = outputs[1][:,:,2].detach()
            logits[batch['labels']<0] = -10

            if lossf:
                lossf = 0.98*lossf+0.02*loss.item()
            else:
                lossf = loss.item()
            tk0.set_postfix(loss = lossf)
            avg_loss += loss.item() / len(train_loader)

            avg_accuracy += torch.mean((torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).to(torch.float) ).item()/len(train_loader)
        tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)
        
        valid_preds1 = np.zeros((len(val_labels),len_val), dtype = np.float32)
        valid_preds2 = np.zeros((len(val_labels),len_val), dtype = np.float32)

        ver_preds1 = np.zeros((len(val_labels)), dtype = np.float32)
        ver_preds2 = np.zeros((len(val_labels)), dtype = np.float32)
        model.half()
        model.eval()
        avg_accuracy = 0.
        tk1 = tqdm(valid_loader,position = 0)
        with torch.no_grad():
            with torch.cuda.amp.autocast():    
                for i,(batch)  in enumerate(tk1):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels']
                    outputs = model(input_ids, attention_mask=attention_mask, labels=None)

                    logits1 = outputs[0][:,:,1].detach()
                    logits1[labels<0] = -10
                    logits2 = outputs[0][:,:,2].detach()
                    logits2[labels<0] = -10                    
                    ver,_ = logits1.max(1)
                    ver_preds1[i*batch_size_test:(i+1)*batch_size_test] = ver.cpu().numpy()                    
                    ver,_ = logits2.max(1) 
                    ver_preds2[i*batch_size_test:(i+1)*batch_size_test] = ver.cpu().numpy() 
                    valid_preds2[i*batch_size_test:(i+1)*batch_size_test,:]=logits2.cpu().numpy()  
                    valid_preds1[i*batch_size_test:(i+1)*batch_size_test,:]=logits1.cpu().numpy()  
                    

        
        valid['ver1'] = ver_preds1
        valid['ver2'] = ver_preds2
        ans = []
        for index, row in tqdm(valid.iterrows(),position = 0):
       
            preds1, = np.where(valid_preds1[index]>0)
            preds2, = np.where(valid_preds2[index]>0)
            preds2.sort()
            max_a = 0
            g_all = []
            for min_a in preds1:
                if max_a> min_a:
                    continue            
                g = ''
                max_a = 0
                for min_b in preds2:
                    if min_b>=min_a:
                        max_a = min_b
                        break
                if max_a == 0 and valid_preds1[index][min_a]>1:
                    max_a = min_a  
                if max_a-min_a > 20:
                    continue
                if max_a>=min_a and min_a>0:
                    k = 0
                      
                    b = np.array(row["val_labels"])

                    s = tokenizer.convert_ids_to_tokens(b[min_a:])
                    for j,w in enumerate(s):
                        
                        if j<=max_a - min_a or "##" in w:
                            g += w + ' '
                        else:    
                            break
                g = g.lower().replace(" ##","").strip()
                g_all.append(g)
            

            g_all = [x for x in set(g_all) if len(x) > 3]      
            ans.append('|'.join(g_all))
        valid["pred"] = ans  
        
        valid["good"] = valid["pred"] == valid["label"]  
 
        avg_accuracy = sum(valid["pred"] ==valid["label"])/len(valid)
        print("Accuracy",avg_accuracy)   
        print(valid[valid.pred!=''][["good","label","pred","ver1","ver2"]].head())
        
        valid[(valid.pred!='') |(valid.label!='') ][["good","label","pred","valid_x"]].to_csv('err.csv', index=False)
        torch.save(model.state_dict(), str(avg_accuracy) + output_model_file)
        generate()
    
if __name__ == '__main__':
    train_files_path = 'train'
    test_files_path = 'test'
    # reading csv files and train & test file paths
    train_df = pd.read_csv('train.csv')
    sample_sub = pd.read_csv('sample_submission.csv')

    sample_sub['text'] = sample_sub['Id'].apply(read_append_return, train_files_path=test_files_path)
    train_df['text'] = train_df['Id'].apply(read_append_return)
    train_df = train_df.drop_duplicates(subset=['Id'])
    #train_df.to_csv('train1.csv', index=False)
    
    new_other = pd.read_csv('data_set1.csv')
    new_other = [clean_text(i) for i in new_other.title.tolist()]
    new_other = list(set(new_other))

    new_other2 = pd.read_csv('new_set.csv')
    new_other2 = [clean_text(i) for i in new_other2.title.tolist() if len(i.split())>2]#
    new_other2 = list(set(new_other2))

    existing_labels = new_other + new_other2

    EPOCHS = 25
    lr = 1e-5
    batch_size = 14
    batch_size_test = 64
    output_model_file = "pytorch_model.bin"
    device = torch.device('cuda')
    BERT_MODEL_PATH = 'bert-base-cased'

    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_PATH,do_lower_case=False)
    model = AutoModelForTokenClassification.from_pretrained(BERT_MODEL_PATH,num_labels=3)
    model.zero_grad()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4,6,8,10,12,14], gamma=0.5)
    Sigmoid = torch.nn.Sigmoid()

    ckpt = torch.load("0.94model.bin",map_location=device)
    model.load_state_dict(ckpt)
    #generate()
    train(train_df)