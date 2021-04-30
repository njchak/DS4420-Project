#!/usr/bin/env python
# coding: utf-8

# In[230]:


# Dependencies
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt


# In[3]:


# Load in data sets
teams = pd.read_csv("ncaam-march-mania-2021/MDataFiles_Stage1/mteams.csv")
tourney_results = pd.read_csv("ncaam-march-mania-2021/MDataFiles_Stage1/MNCAATourneyCompactResults.csv")
tourney_results_detailed = pd.read_csv("ncaam-march-mania-2021/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv")
seeds = pd.read_csv("ncaam-march-mania-2021/MDataFiles_Stage1/MNCAATourneySeeds.csv")
regular_season = pd.read_csv("ncaam-march-mania-2021/MDataFiles_Stage1/MRegularSeasonCompactResults.csv")
regular_season_detailed = pd.read_csv("ncaam-march-mania-2021/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv")


# In[4]:


# Inspect data
regular_season


# In[5]:


# Format seeds to be a number only
seeds['Seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
seeds


# In[6]:


# Aggregate by season for wins and points
wins = regular_season.groupby(['Season', 'WTeamID'])['WTeamID'].count()
loses = regular_season.groupby(['Season', 'LTeamID'])['LTeamID'].count()
points_w = regular_season.groupby(['Season', 'WTeamID'])['WScore'].mean()
points_l = regular_season.groupby(['Season', 'LTeamID'])['LScore'].mean()

wins.index = wins.index.rename(['Season', 'TeamID'])
loses.index = loses.index.rename(['Season', 'TeamID'])
points_w.index = points_w.index.rename(['Season', 'TeamID'])
points_l.index = points_l.index.rename(['Season', 'TeamID'])


# In[8]:


# Get records, PPG for each team each season
records = pd.merge(wins.reset_index(), loses.reset_index())
records = records.rename(columns={'WTeamID': 'Wins', 'LTeamID': 'Loses'})
records['win_percentage'] = records['Wins'] / (records['Wins'] + records['Loses'])
records = records.merge(points_w.reset_index()).merge(points_l.reset_index())
total_games = records['Wins'] + records['Loses']
records['ppg'] = (records['Wins'] / total_games) * records['WScore'] + (records['Loses'] / total_games) * records['LScore']
records = records.drop(columns = ['WScore', 'LScore'])
records


# In[10]:


# Combine with tournament data
combined_df = pd.merge(tourney_results, records, left_on = ['Season', 'WTeamID'], right_on = ['Season', 'TeamID'])
combined_df = combined_df.rename(columns={'Wins': 'WTeamSeasonWins', 'Loses': 'WTeamSeasonLoses', 
                                    'win_percentage': 'WTeamWinPercentage', 'ppg': 'WTeamPPG'})
combined_df = pd.merge(combined_df, records, left_on = ['Season', 'LTeamID'], right_on = ['Season', 'TeamID'])
combined_df = combined_df.rename(columns={'Wins': 'LTeamSeasonWins', 'Loses': 'LTeamSeasonLoses', 
                                    'win_percentage': 'LTeamWinPercentage', 'ppg': 'LTeamPPG'})
combined_df = combined_df.drop(columns = ['TeamID_x', 'TeamID_y'])
combined_df = pd.merge(combined_df, seeds, left_on = ['Season', 'WTeamID'], right_on = ['Season', 'TeamID'])
combined_df = combined_df.drop(columns = ['TeamID'])
combined_df = combined_df.rename(columns={'Seed': 'WSeed'})
combined_df = pd.merge(combined_df, seeds, left_on = ['Season', 'LTeamID'], right_on = ['Season', 'TeamID'])
combined_df = combined_df.drop(columns = ['TeamID'])
combined_df = combined_df.rename(columns={'Seed': 'LSeed'})
combined_df


# In[11]:


# Convert win / loss to higher and lower seed, since game outcome is not determined until after playtime
upsets = combined_df[combined_df['WSeed'] > combined_df['LSeed']]
upsets = upsets.rename(columns = {'LTeamID': 'HighSeedTeamID', 'WTeamID': 'LowSeedTeamID',
                                       'LTeamSeasonWins': 'HighSeedSeasonWins', 'WTeamSeasonWins': 'LowSeedSeasonWins',
                                       'LTeamSeasonLoses': 'HighSeedSeasonLoses', 'WTeamSeasonLoses': 'LowSeedSeasonLoses',
                                       'LTeamWinPercentage': 'HighSeedWinPercentage', 'WTeamWinPercentage': 'LowSeedWinPercentage',
                                       'LTeamPPG': 'HighSeedPPG', 'WTeamPPG': 'LowSeedPPG',
                                       'LSeed': 'HighSeed', 'WSeed': 'LowSeed'})
upsets = upsets.drop(columns = ['WScore', 'LScore', 'WLoc', 'NumOT'])
upsets['HighSeedWon'] = 0
upsets


# In[12]:


# Same as above cell
favorites = combined_df[combined_df['LSeed'] >= combined_df['WSeed']]
favorites = favorites.rename(columns = {'WTeamID': 'HighSeedTeamID', 'LTeamID': 'LowSeedTeamID',
                                       'WTeamSeasonWins': 'HighSeedSeasonWins', 'LTeamSeasonWins': 'LowSeedSeasonWins',
                                       'WTeamSeasonLoses': 'HighSeedSeasonLoses', 'LTeamSeasonLoses': 'LowSeedSeasonLoses',
                                       'WTeamWinPercentage': 'HighSeedWinPercentage', 'LTeamWinPercentage': 'LowSeedWinPercentage',
                                       'WTeamPPG': 'HighSeedPPG', 'LTeamPPG': 'LowSeedPPG',
                                       'WSeed': 'HighSeed', 'LSeed': 'LowSeed'})
favorites = favorites.drop(columns = ['WScore', 'LScore', 'WLoc', 'NumOT'])
favorites['HighSeedWon'] = 1
favorites


# In[13]:


# Combine for training data
train_historic = pd.concat([favorites, upsets])


# In[270]:


# Break DF into X and Y, and check shape to determine dimensions for neural network
X = train_historic[train_historic.columns[:-1]]
Y = train_historic[train_historic.columns[-1]]
X.shape


# In[274]:


# use first 25 years for training, and last 10 for testing for about a 70/30 split
training_data = train_historic[train_historic['Season'] < 2010]
testing_data = train_historic[train_historic['Season'] >= 2010]

X_train = training_data[training_data.columns[:-1]]
Y_train = training_data[training_data.columns[-1]]
X_test = testing_data[testing_data.columns[:-1]]
Y_test = testing_data[testing_data.columns[-1]]


# In[271]:


# Defining model with help from: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# Use batch normalization layer for normalizing data, and dropout layers to reduce model bias
def define_model(input_df):
    model = Sequential();
    model.add(BatchNormalization(epsilon=0.001))
    model.add(Dense(256, input_dim = input_df.shape[1], activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model


# In[251]:


nn = define_model(X_train)


# In[252]:


history = nn.fit(X_train, Y_train, epochs = 256, batch_size = 32)


# In[310]:


def plot_model_accuracy_and_loss(history, accuracy_name, loss_name):
    plt.plot(history.history['accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.savefig(accuracy_name, dpi=300, bbox_inches='tight')
    plt.clf()
    
    plt.plot(history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss Over Epochs')
    plt.savefig(loss_name, dpi=300, bbox_inches='tight')
    plt.clf()
    
    
plot_model_accuracy_and_loss(history, 'model_accuracy_v1.png', 'model_loss_v1.png')


# In[249]:


# Try with 512 epochs
nn2 = define_model(X_train)
history512 = nn.fit(X_train, Y_train, epochs = 512, batch_size = 32)


# In[254]:


# Try with 128 epochs
nn3 = define_model(X_train)
history128 = nn3.fit(X_train, Y_train, epochs = 128, batch_size = 32)


# In[311]:


plot_model_accuracy_and_loss(history512, 'model_accuracy_v1_512.png', 'model_loss_v1_512.png')


# In[312]:


plot_model_accuracy_and_loss(history128, 'model_accuracy_v1_128.png', 'model_loss_v1_128.png')


# In[293]:


# Will use 256 epochs going forward. 
predictions = (nn.predict(X_test, batch_size = 32) > 0.5).astype("int32")
mean_squared_error(Y_test, predictions)


# In[294]:


accuracy_score(Y_test, predictions)


# In[295]:


confusion_matrix(Y_test, predictions)


# In[296]:


accuracy = nn.evaluate(X_train, Y_train)
print('Accuracy: ' + str(accuracy[1]))


# In[79]:


# Aggregate for detailed statistics; quite ugly but gets job done
wins = regular_season_detailed.groupby(['Season', 'WTeamID'])['WTeamID'].count()
loses = regular_season_detailed.groupby(['Season', 'LTeamID'])['LTeamID'].count()
points_w = regular_season_detailed.groupby(['Season', 'WTeamID'])['WScore'].mean()
points_l = regular_season_detailed.groupby(['Season', 'LTeamID'])['LScore'].mean()
fgm_w = regular_season_detailed.groupby(['Season', 'WTeamID'])['WFGM'].mean()
fgm_l = regular_season_detailed.groupby(['Season', 'LTeamID'])['LFGM'].mean()
fga_w = regular_season_detailed.groupby(['Season', 'WTeamID'])['WFGA'].mean()
fga_l = regular_season_detailed.groupby(['Season', 'LTeamID'])['LFGA'].mean()
fgm3_w = regular_season_detailed.groupby(['Season', 'WTeamID'])['WFGA3'].mean()
fgm3_l = regular_season_detailed.groupby(['Season', 'LTeamID'])['LFGA3'].mean()
ftm_w = regular_season_detailed.groupby(['Season', 'WTeamID'])['WFTM'].mean()
ftm_l = regular_season_detailed.groupby(['Season', 'LTeamID'])['LFTM'].mean()
fta_w = regular_season_detailed.groupby(['Season', 'WTeamID'])['WFTA'].mean()
fta_l = regular_season_detailed.groupby(['Season', 'LTeamID'])['LFTA'].mean()
or_w = regular_season_detailed.groupby(['Season', 'WTeamID'])['WOR'].mean()
or_l = regular_season_detailed.groupby(['Season', 'LTeamID'])['LOR'].mean()
dr_w = regular_season_detailed.groupby(['Season', 'WTeamID'])['WDR'].mean()
dr_l = regular_season_detailed.groupby(['Season', 'LTeamID'])['LDR'].mean()
ast_w = regular_season_detailed.groupby(['Season', 'WTeamID'])['WAst'].mean()
ast_l = regular_season_detailed.groupby(['Season', 'LTeamID'])['LAst'].mean()
to_w = regular_season_detailed.groupby(['Season', 'WTeamID'])['WTO'].mean()
to_l = regular_season_detailed.groupby(['Season', 'LTeamID'])['LTO'].mean()
stl_w = regular_season_detailed.groupby(['Season', 'WTeamID'])['WStl'].mean()
stl_l = regular_season_detailed.groupby(['Season', 'LTeamID'])['LStl'].mean()
blk_w = regular_season_detailed.groupby(['Season', 'WTeamID'])['WBlk'].mean()
blk_l = regular_season_detailed.groupby(['Season', 'LTeamID'])['LBlk'].mean()
pf_w = regular_season_detailed.groupby(['Season', 'WTeamID'])['WPF'].mean()
pf_l = regular_season_detailed.groupby(['Season', 'LTeamID'])['LPF'].mean()

dataframes = [wins, loses, points_w, points_l, fgm_w, fgm_l, fga_w, fga_l, fgm3_w, fgm3_l, ftm_w, ftm_l,
             fta_w, fta_l, or_w, or_l, dr_w, dr_l, ast_w, ast_l, to_w, to_l, stl_w, stl_l, blk_w, blk_l, 
             pf_w, pf_l]

for df in dataframes:
    df.index = df.index.rename(['Season', 'TeamID'])


# In[96]:


records_v2 = wins.to_frame().reset_index()
for df in dataframes:
    records_v2 = records_v2.merge(df.to_frame().reset_index())

records_v2 = records_v2.rename(columns={'WTeamID': 'Wins', 'LTeamID': 'Loses'})

total_games = records_v2['Wins'] + records_v2['Loses']

column_pairs = [('WScore', 'LScore'), ('WFGM', 'LFGM'), ('WFGA', 'LFGA'), ('WFGA3', 'LFGA3'), ('WFTM', 'LFTM'),
               ('WFTA', 'LFTA'), ('WOR', 'LOR'), ('WDR', 'LDR'), ('WAst', 'LAst'), ('WTO', 'LTO'), ('WStl', 'LStl'),
               ('WBlk', 'LBlk'), ('WPF', 'LPF')]
new_col_name = ['ppg', 'fgm', 'fga', 'fga3', 'ftm', 'fta', 'or', 'dr', 'ast', 'to', 'stl', 'blk', 'pf']


for i, col in enumerate(column_pairs):
    records_v2[new_col_name[i]] = (records_v2['Wins'] / total_games) * records_v2[column_pairs[i][0]] + (records_v2['Loses'] / total_games) * records_v2[column_pairs[i][1]]
    records_v2 = records_v2.drop(columns = list(column_pairs[i]))


# In[104]:


# Follows same logic as first dataset at this point
combined_df_v2 = pd.merge(tourney_results, records_v2, left_on = ['Season', 'WTeamID'], right_on = ['Season', 'TeamID'])
combined_df_v2 = pd.merge(combined_df_v2, records_v2, left_on = ['Season', 'LTeamID'], right_on = ['Season', 'TeamID'])
combined_df_v2 = pd.merge(combined_df_v2, seeds, left_on = ['Season', 'WTeamID'], right_on = ['Season', 'TeamID'])
combined_df_v2 = pd.merge(combined_df_v2, seeds, left_on = ['Season', 'LTeamID'], right_on = ['Season', 'TeamID'])
combined_df_v2 = combined_df_v2.rename(columns={'Seed': 'LSeed'})
combined_df_v2


# In[257]:


combined_df_v2.columns


# In[265]:


upsets_v2 = combined_df_v2[combined_df_v2['Seed_x'] > combined_df_v2['Seed_y']]
upsets_v2 = upsets_v2.rename(columns = {'LTeamID': 'HighSeedTeamID', 'WTeamID': 'LowSeedTeamID',
                                       'Wins_x': 'LowSeedSeasonWins', 'Loses_x': 'LowSeedSeasonLoses', 
                                        'ppg_x': 'LowSeedPPG', 'fgm_x': 'LowSeedFGM', 'fga_x': 'LowSeedFGA',
                                       'fga3_x': 'LowSeedFGA3', 'ftm_x': 'LowSeedFTM', 
                                        'fta_x': 'LowSeedFTA', 'or_x': 'LowSeedOR', 'dr_x': 'LowSeedDR', 
                                        'ast_x': 'LowSeedAST', 'to_x': 'LowSeedTO', 'stl_x': 'LowSeedSTL',
                                       'blk_x': 'LowSeedBLK', 'pf_x': 'LowSeedPF', 
                                        'Wins_y': 'HighSeedSeasonWins', 'Loses_y': 'HighSeedSeasonLoses', 
                                        'ppg_y': 'HighSeedPPG', 'fgm_y': 'HighSeedFGM',
                                       'fga_y': 'HighSeedFGA', 'fga3_y': 'HighSeedFGA3', 
                                        'ftm_y': 'HighSeedFTM' , 'fta_y': 'HighSeedFTA', 
                                        'or_y': 'HighSeedOR', 'dr_y': 'HighSeedDR', 'ast_y': 'HighSeedAST', 
                                        'to_y': 'HighSeedTO', 'stl_y': 'HighSeedSTL', 'blk_y': 'HighSeedBLK', 
                                        'pf_y': 'HighSeedPF', 'Seed_x': 'LowSeed', 'Seed_y': 'HighSeed'})
upsets_v2 = upsets_v2.drop(columns = ['WScore', 'LScore', 'WLoc', 'NumOT', 'TeamID_x', 'TeamID_y'])
upsets_v2['HighSeedWon'] = 0
upsets_v2


# In[266]:


favorites_v2 = combined_df_v2[combined_df_v2['Seed_x'] <= combined_df_v2['Seed_y']]
favorites_v2 = favorites_v2.rename(columns = {'LTeamID': 'LowSeedTeamID', 'WTeamID': 'HighSeedTeamID',
                                       'Wins_y': 'LowSeedSeasonWins', 'Loses_y': 'LowSeedSeasonLoses', 
                                        'ppg_y': 'LowSeedPPG', 'fgm_y': 'LowSeedFGM', 'fga_y': 'LowSeedFGA',
                                       'fga3_y': 'LowSeedFGA3', 'ftm_y': 'LowSeedFTM', 
                                        'fta_y': 'LowSeedFTA', 'or_y': 'LowSeedOR', 'dr_y': 'LowSeedDR', 
                                        'ast_y': 'LowSeedAST', 'to_y': 'LowSeedTO', 'stl_y': 'LowSeedSTL',
                                       'blk_y': 'LowSeedBLK', 'pf_y': 'LowSeedPF', 
                                        'Wins_x': 'HighSeedSeasonWins', 'Loses_x': 'HighSeedSeasonLoses', 
                                        'ppg_x': 'HighSeedPPG', 'fgm_x': 'HighSeedFGM',
                                       'fga_x': 'HighSeedFGA', 'fga3_x': 'HighSeedFGA3', 
                                        'ftm_x': 'HighSeedFTM' , 'fta_x': 'HighSeedFTA', 
                                        'or_x': 'HighSeedOR', 'dr_x': 'HighSeedDR', 'ast_x': 'HighSeedAST', 
                                        'to_x': 'HighSeedTO', 'stl_x': 'HighSeedSTL', 'blk_x': 'HighSeedBLK', 
                                        'pf_x': 'HighSeedPF', 'Seed_y': 'LowSeed', 'Seed_x': 'HighSeed'})
favorites_v2 = favorites_v2.drop(columns = ['WScore', 'LScore', 'WLoc', 'NumOT', 'TeamID_x', 'TeamID_y'])
favorites_v2['HighSeedWon'] = 1
favorites_v2


# In[302]:


# More detailed regular season stats, starting in 2003 season
train_detailed = pd.concat([favorites_v2, upsets_v2])


# In[303]:


# Break DF into X and Y, and check shape to determine dimensions for neural network
X = train_detailed[train_detailed.columns[:-1]]
Y = train_detailed[train_detailed.columns[-1]]
X.shape


# In[304]:


# use first 12 years for training, and last 5 for testing for about a 70/30 split
training_data_v2 = train_detailed[train_detailed['Season'] < 2015]
testing_data_v2 = train_detailed[train_detailed['Season'] >= 2015]

X_train_v2 = training_data_v2[training_data_v2.columns[:-1]]
Y_train_v2 = training_data_v2[training_data_v2.columns[-1]]
X_test_v2 = testing_data_v2[testing_data_v2.columns[:-1]]
Y_test_v2 = testing_data_v2[testing_data_v2.columns[-1]]


# In[305]:


nn_v2 = define_model(X_train_v2)
nn_v2_128 = define_model(X_train_v2)
nn_v2_512 = define_model(X_train_v2)


# In[306]:


history_v2 = nn_v2.fit(X_train_v2, Y_train_v2, epochs = 256, batch_size = 32)


# In[307]:


history_v2_128 = nn_v2_128.fit(X_train_v2, Y_train_v2, epochs = 128, batch_size = 32)


# In[308]:


history_v2_512 = nn_v2_512.fit(X_train_v2, Y_train_v2, epochs = 512, batch_size = 32)


# In[309]:


plot_model_accuracy_and_loss(history_v2, 'model_accuracy_v2_256.png', 'model_loss_v2_256.png')
plot_model_accuracy_and_loss(history_v2_128, 'model_accuracy_v2_128.png', 'model_loss_v2_128.png')
plot_model_accuracy_and_loss(history_v2_512, 'model_accuracy_v2_512.png', 'model_loss_v2_512.png')


# In[299]:


# Will use 256 epochs going forward. 
predictions = (nn_v2.predict(X_test_v2, batch_size = 32) > 0.5).astype("int32")
mean_squared_error(Y_test_v2, predictions)


# In[300]:


accuracy_score(Y_test_v2, predictions)


# In[301]:


confusion_matrix(Y_test_v2, predictions)

