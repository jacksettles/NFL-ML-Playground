#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from math import ceil
import os


# # Read in data and preprocess it

# In[3]:


NFL = pd.read_csv(r"NFL Play by Play 2009-2018 (v5).csv", low_memory=False)


# In[4]:


''' Filter rows '''
NFL = NFL[(NFL.down.isin([1, 2, 3, 4])) & ((NFL.play_type == 'run') | (NFL.play_type == 'pass'))
          & (NFL.incomplete_pass == 0)]


# In[5]:


NFL = NFL[['game_id', 'posteam', 'posteam_type', 'defteam', 'game_seconds_remaining', 'yardline_100', 'down', 'ydstogo', 'yards_gained', 'score_differential',
           'play_type', 'pass_length', 'pass_location', 'run_location', 'run_gap', 'shotgun', 'no_huddle']]
NFL.dropna(subset=['yards_gained'], inplace=True)
NFL.dropna(subset=['game_seconds_remaining'], inplace=True)
NFL['game_seconds_remaining'] = NFL['game_seconds_remaining'].astype(int)
NFL['yards_gained'] = NFL['yards_gained'].astype(int)


# In[6]:


def get_play(play):
    play_type, pass_length, pass_location, run_location, run_gap = play
    if play_type == 'pass':
        if (pass_length != 'unknown') & (pass_location != 'unknown'):
            return play_type.capitalize() + " " + pass_length.capitalize() + " " + pass_location.capitalize()
    elif play_type == 'run':
        if run_location == 'middle':
            return "Run Middle"
        elif (run_location != 'unknown') & (run_gap != 'unknown'):
            return "Run " + run_location.capitalize() + " " + run_gap.capitalize()


# In[7]:


# Sit is situation
# Down, yards to go, and yards gained gets fed in to determine play success (1), which is defined below
def play_success(sit):
    down, ydstogo, yards_gained = sit
    if down == 1:
        if (yards_gained/ydstogo) >= 0.5:
            return 1
        else:
            return 0
    elif down == 2:
        if (yards_gained/ydstogo) >= 0.7:
            return 1
        else:
            return 0
    else:
        if (yards_gained/ydstogo) >= 1:
            return 1
        else:
            return 0


# In[8]:


''' Turn unknown values into "unknown" '''
NFL = NFL.replace(np.nan, 'unknown', regex=True)


# In[9]:


''' Create column in DF for complete play type '''
NFL['complete_play'] = NFL[['play_type', 'pass_length', 'pass_location', 'run_location', 'run_gap']].apply(get_play, axis=1)


# In[10]:


cp = NFL.groupby('complete_play').count()[['play_type']]
# print(cp)

''' Get the sum of all the plays '''
total = 0
for x in cp['play_type']:
    total += x

''' Turn each value into a percentage of the sum '''
for i, row in cp.iterrows():
    cp.at[i, 'play_type'] *= 100 / total
    
# print(cp)


# In[11]:


NFL['complete_play'] = NFL['complete_play'].replace('', np.nan, regex=True)
NFL.dropna(subset=['complete_play'], inplace=True)


# In[12]:


''' Create column for successful plays '''
NFL['play_success'] = NFL[['down', 'ydstogo', 'yards_gained']].apply(play_success, axis=1)


# In[13]:


''' Labeling play types to numerical values
    Pass Deep Left->0
    Pass Deep Middle->1
    Pass Deep Right->2
    Pass Short Left->3
    Pass Short Middle->4
    Pass Short Right->5
    Run Left End->6
    Run Left Guard->7
    Run Left Tackle->8
    Run Middle->9
    Run Right End->10
    Run Right Guard->11
    Run Right Tackle->12 '''
encoder = LabelEncoder()
NFL['cp_label'] = encoder.fit_transform(NFL['complete_play'])


# In[14]:


'''Yardage to go is binned'''
def yardage_bin(y):
    if y <= 5:
        return 0
    elif (y > 5) & (y <= 9):
        return 1
    elif (y > 9) & (y <= 15):
        return 2
    else:
        return 3


# In[23]:


'''Score differential is binned
The value equivalent to the number of possesions
the posessing team is winning or losing by'''
def score_diff_bin(x: int) -> int:
    if x == 0:
        return 0
    sign: int = abs(x) / x
    return sign * min(ceil(abs(x) / 8),4)


# In[24]:


''' Replace the ydstogo values with their bin numbers '''
NFL['ytg_bin'] = [yardage_bin(x) for x in NFL['ydstogo']]


# In[25]:


''' Replace score differential values with bin numbers'''
NFL['score_diff_bin'] = [score_diff_bin(x) for x in NFL['score_differential']]


# In[17]:


''' Make yardline_100 and game_seconds_remaining on a scale of 0 to 1 '''
NFL['yardline_100'] /= 100
NFL['game_seconds_remaining'] /= 3600


# In[18]:


''' One hot encoding downs, ytg bins, cp_label, and score diff bins'''
one_hot_downs = pd.get_dummies(NFL['down'], prefix='down')
one_hot_ytg = pd.get_dummies(NFL['ytg_bin'], prefix='ytg_bin')
one_hot_cpl = pd.get_dummies(NFL['cp_label'], prefix='cpl')
one_hot_sd = pd.get_dummies(NFL['score_diff_bin'], prefix='score_diff_bin')
one_hot_posteam = pd.get_dummies(NFL['posteam'], prefix = 'posteam')
one_hot_posteam_type = pd.get_dummies(NFL['posteam_type'], prefix = 'posteam_type')
one_hot_defteam = pd.get_dummies(NFL['defteam'], prefix = 'defteam')


# In[19]:


''' Add one hot values to DF '''
NFL = pd.concat([NFL, one_hot_downs, one_hot_posteam,
                 one_hot_posteam_type, one_hot_defteam,
                 one_hot_ytg, one_hot_cpl, one_hot_sd], axis=1)


# # Split the data up between input features, targets, and then train and test sets

# In[20]:


x = NFL[['game_seconds_remaining', 'yardline_100', 'down_1.0', 'down_2.0', 'down_3.0', 'down_4.0', 'ytg_bin_0',
         'ytg_bin_1', 'ytg_bin_2', 'ytg_bin_3', 'cpl_0', 'cpl_1', 'cpl_2', 'cpl_3', 'cpl_4', 'cpl_5', 'cpl_6', 'cpl_7',
         'cpl_8', 'cpl_9', 'cpl_10', 'cpl_11', 'cpl_12', 'score_diff_bin_-4', 'score_diff_bin_-3', 'score_diff_bin_-2',
         'score_diff_bin_-1', 'score_diff_bin_0', 'score_diff_bin_1', 'score_diff_bin_2', 'score_diff_bin_3',
         'score_diff_bin_4', 'posteam_ARI', 'posteam_ATL', 'posteam_BAL', 'posteam_BUF', 'posteam_CAR', 'posteam_CHI',
         'posteam_CIN', 'posteam_CLE', 'posteam_DAL', 'posteam_DEN', 'posteam_DET', 'posteam_GB', 'posteam_HOU',
         'posteam_IND', 'posteam_JAC', 'posteam_JAX', 'posteam_KC', 'posteam_LA', 'posteam_LAC', 'posteam_MIA',
         'posteam_MIN', 'posteam_NE', 'posteam_NO', 'posteam_NYG', 'posteam_NYJ', 'posteam_OAK', 'posteam_PHI',
         'posteam_PIT', 'posteam_SD', 'posteam_SEA', 'posteam_SF', 'posteam_STL', 'posteam_TB', 'posteam_TEN',
         'posteam_WAS', 'posteam_type_away', 'posteam_type_home', 'defteam_ARI', 'defteam_ATL', 'defteam_BAL',
         'defteam_BUF', 'defteam_CAR', 'defteam_CHI', 'defteam_CIN', 'defteam_CLE', 'defteam_DAL', 'defteam_DEN',
         'defteam_DET', 'defteam_GB', 'defteam_HOU', 'defteam_IND', 'defteam_JAC', 'defteam_JAX', 'defteam_KC', 
         'defteam_LA', 'defteam_LAC', 'defteam_MIA', 'defteam_MIN', 'defteam_NE', 'defteam_NO', 'defteam_NYG', 'defteam_NYJ',
         'defteam_OAK', 'defteam_PHI', 'defteam_PIT', 'defteam_SD', 'defteam_SEA', 'defteam_SF', 'defteam_STL', 
         'defteam_TB', 'defteam_TEN', 'defteam_WAS']]

y = NFL['play_success']

train_x, test_x, train_y, test_y = train_test_split(x, y, shuffle=True, random_state=0)


# In[ ]:





# In[21]:


def decision_tree():
    DT = DecisionTreeClassifier(max_depth=10, random_state=1)
    DT.fit(train_x, train_y)
    predictions = DT.predict(test_x)
    DTscore = accuracy_score(test_y, predictions)
    print("Accuracy using Decision Tree: " + str(DTscore))
    print("")


# In[22]:


def random_forest():
    RF = RandomForestClassifier(max_depth=10, n_estimators=25, random_state=1)
    RF.fit(train_x, train_y)
    predictions = RF.predict(test_x)
    RFscore = accuracy_score(test_y, predictions)
    print("Accuracy using Random Forest: " + str(RFscore))
    print("")


# In[23]:


def nearest_neighbor():
    knc = KNeighborsClassifier(n_neighbors=3)
    knc.fit(train_x, train_y)
    predictions = knc.predict(test_x)
    knscore = accuracy_score(test_y, predictions)
    print("Accuracy using k-Nearest Neighbors: " + str(knscore))
    print("")


# In[24]:


def neural_network():
    nn = MLPClassifier(verbose=True, random_state=1, early_stopping=True, max_iter=300)
    nn.fit(train_x, train_y)
    predictions = nn.predict(test_x)
    nnScore = accuracy_score(test_y, predictions)
    print("Accuracy using Neural Network: " + str(nnScore))


# # PyTorch!

# In[25]:


# Hyper Parameters 
input_size = 104
num_classes = 2
num_epochs = 5
learning_rate = 0.001
batch_size = 64


# In[26]:


if torch.cuda.is_available():
    # CUDA is available, you can proceed to use it
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
else:
    # CUDA is not available, use CPU
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU.')


# In[27]:


torch_NFL = NFL[['game_seconds_remaining', 'yardline_100', 'down_1.0', 'down_2.0', 'down_3.0', 'down_4.0', 'ytg_bin_0',
         'ytg_bin_1', 'ytg_bin_2', 'ytg_bin_3', 'cpl_0', 'cpl_1', 'cpl_2', 'cpl_3', 'cpl_4', 'cpl_5', 'cpl_6', 'cpl_7',
         'cpl_8', 'cpl_9', 'cpl_10', 'cpl_11', 'cpl_12', 'score_diff_bin_-4', 'score_diff_bin_-3', 'score_diff_bin_-2',
         'score_diff_bin_-1', 'score_diff_bin_0', 'score_diff_bin_1', 'score_diff_bin_2', 'score_diff_bin_3',
         'score_diff_bin_4', 'posteam_ARI', 'posteam_ATL', 'posteam_BAL', 'posteam_BUF', 'posteam_CAR', 'posteam_CHI',
         'posteam_CIN', 'posteam_CLE', 'posteam_DAL', 'posteam_DEN', 'posteam_DET', 'posteam_GB', 'posteam_HOU',
         'posteam_IND', 'posteam_JAC', 'posteam_JAX', 'posteam_KC', 'posteam_LA', 'posteam_LAC', 'posteam_MIA',
         'posteam_MIN', 'posteam_NE', 'posteam_NO', 'posteam_NYG', 'posteam_NYJ', 'posteam_OAK', 'posteam_PHI',
         'posteam_PIT', 'posteam_SD', 'posteam_SEA', 'posteam_SF', 'posteam_STL', 'posteam_TB', 'posteam_TEN',
         'posteam_WAS', 'posteam_type_away', 'posteam_type_home', 'defteam_ARI', 'defteam_ATL', 'defteam_BAL',
         'defteam_BUF', 'defteam_CAR', 'defteam_CHI', 'defteam_CIN', 'defteam_CLE', 'defteam_DAL', 'defteam_DEN',
         'defteam_DET', 'defteam_GB', 'defteam_HOU', 'defteam_IND', 'defteam_JAC', 'defteam_JAX', 'defteam_KC', 
         'defteam_LA', 'defteam_LAC', 'defteam_MIA', 'defteam_MIN', 'defteam_NE', 'defteam_NO', 'defteam_NYG', 'defteam_NYJ',
         'defteam_OAK', 'defteam_PHI', 'defteam_PIT', 'defteam_SD', 'defteam_SEA', 'defteam_SF', 'defteam_STL', 
         'defteam_TB', 'defteam_TEN', 'defteam_WAS', 'play_success']]


# In[28]:


torch_train, torch_test = train_test_split(torch_NFL, test_size = 0.2, random_state = 42)


# In[29]:


class NFL_Data(Dataset):
    
    def __init__(self, df, transform = transforms.ToTensor()):
        self.data = df
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        
        features = sample.iloc[:-1].values
        target = sample.iloc[-1]
        
        features = np.array(list(features))
        target = np.array(target)
        
        #print("Features: ", type(features))
        #print("Target data type: ", type(target))
        
        features = torch.tensor(features)
        target = torch.tensor(target)
        #if self.transform:
            #features = self.transform(features)
            
        return features, target


# In[30]:


train_dataset = NFL_Data(torch_test)
test_dataset = NFL_Data(torch_train)


# In[31]:


train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)


# In[25]:


class PlayPredictor(nn.Module):
    
    def __init__(self, input_size, num_classes):
        super(PlayPredictor, self).__init__()
        self.linear1 = nn.Linear(input_size, 84)
        self.linear2 = nn.Linear(84, 48)
        self.linear3 = nn.Linear(48, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, 3)

        
    def forward(self, x):
        leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        x = leaky_relu(self.linear1(x))
        x = leaky_relu(self.linear2(x))
        x = leaky_relu(self.linear3(x))
        x = leaky_relu(self.linear4(x))
        x = self.linear5(x)
        return x


# In[26]:


def train(train_loader, test_loader, model_name="Torch-Play-Success", save_dir="model"):
    model = PlayPredictor(input_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # If there is no save directory yet, make one
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model = model.to(device)
    
    #outer training loop goes through all training data 5 times (num_epochs)
    for epoch in range(1, num_epochs+1):
        for step, (features, labels) in enumerate(train_loader):
            model.train()

            features = features.to(device)
            labels = labels.to(device)
            
            features = features.view(-1, input_size).float()

            #zero out the gradient before each batch is processed
            optimizer.zero_grad()

            #pass the batch of images through the model and collect the outputs
            #outputs is 100 x 10 
            outputs = model(features)

            #calculate the loss for the mini-batch using the model-predicted scores over labels
            #and the known labels; labels is a tensor of size 100
            train_loss = criterion(outputs, labels.to(torch.long))

            #propogate the loss backwards through the network
            train_loss.backward()

            #take a step with the optimzer to update parameters
            optimizer.step()

            #progress report
            if (step) % 50 == 0:
                
                # Take model's predictions on training set
                # Calculate its accuracy on training set
                train_correct = 0
                train_total = 0
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum()
                train_acc = (100 * (train_correct/train_total))
                train_acc = train_acc.item()
                
                # Now test the model on the test set, get the test accuracy and test loss
                accuracy, test_loss = test(model=model, test_loader=test_loader, loss_func=criterion)

                # Show the results
                print('Epoch[{}]:Step[{}] Train Loss: {:.2f}\tTrain Acc: {:.2f}%\tTest Loss: {:.2f}\t\tTest Acc: {:.2f}%'.format(
                    epoch, step, train_loss, train_acc, test_loss, accuracy))
                save_model(model, model_name, accuracy)
        scheduler.step()


# In[27]:


def test(model=None, test_image=None, test_loader=None, loss_func=None):
    correct = 0
    total = 0
    
    model = model.to(device)
    
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            
            inputs, labels = data
            
            inputs, labels = inputs.to(device).float(), labels.to(device).long()

            inputs = inputs.view(-1, input_size)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            test_loss = loss_func(outputs, labels)

            # Increment total number of observations seen by
            # number of items in this batch
            total += labels.size(0)

            # Increment total number of correct predictions by
            # number of correct predictions in this batch
            correct += (predicted == labels).sum()

        accuracy = (100 * (correct/total))
        accuracy = accuracy.item()
        return accuracy, test_loss


# In[41]:


def save_highest_accuracy(accuracy):
    with open("highest_accuracy.txt", "w") as f:
        f.write(str(accuracy))


# In[42]:


def save_model(model, model_name, accuracy):

    highest_accuracy = load_highest_accuracy()

    # Only save the model if its accuracy is higher than the previous model's
    if accuracy > highest_accuracy:

        save_highest_accuracy(accuracy)

        file_path = "./model/{}.pt".format(model_name)

        print("New highest accuracy. Saving model ...")
        print()

        torch.save(model.state_dict(), file_path)


# In[43]:


def load_highest_accuracy():
    if os.path.exists("highest_accuracy.txt"):
        with open("highest_accuracy.txt", "r") as f:
            return float(f.read())
    else:
        return 0.0  # Default to 0.0 if the file doesn't exist


# In[ ]:





# In[ ]:





# In[ ]:




