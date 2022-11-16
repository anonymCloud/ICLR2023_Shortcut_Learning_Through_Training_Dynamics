import seaborn as sns
import matplotlib.pyplot as plt  
from tqdm import tqdm   
import pandas as pd
import torch


# df is pandas dataframe (it has 0 and 1 labels for each column)
def analyse_correlations(df, disease, shortcut):
    
    a = len(df[(df[disease]==1)&(df[shortcut]==0)])
    b = len(df[(df[disease]==1)&(df[shortcut]==1)])
    c = len(df[(df[disease]==0)&(df[shortcut]==0)])
    d = len(df[(df[disease]==0)&(df[shortcut]==1)])

    ax= plt.subplot()
    sns.heatmap([[a,b],[c,d]], annot=True, fmt='g', ax=ax, cmap='coolwarm')  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel(shortcut);ax.set_ylabel(disease)
    ax.set_title('Correlation Analysis')
    ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['1', '0'])
    plt.show()


# takes in a list and plots grid of images
def plot_images(img_list, rows, cols):
    plt.figure(figsize=(20,20))
    for i in range(rows*cols):
        plt.subplot(rows,cols,i+1)
        img = plt.imread(img_list[i])
        plt.imshow(img)
    plt.show()

# search for images in the dataset with the given conditions
def viz_dataset(df, conditions, path_col='path'):
    bool_arr = (df[conditions[0]['name']]==conditions[0]['value'])
    for cond in conditions[1:]:
        bool_arr = (bool_arr) & (df[cond['name']]==cond['value'])
    df2 = df[bool_arr]
    df2 = df2.sample(frac=1)
    plot_images(list(df2.head(9)[path_col]),3,3)



# either df or path should be provided. path overrides df
# provide split info if you're only interested in subset of data
'''
Usage:
spurious_corr = {'core':['Pneumothorax',0.0,1.0], 'spurious':['age_in_years',50.0]}
split_info = {'col_name': 'val_train_split', 'values':[0,1]}

df_shortcut, groups = generate_csv(spurious_corr, path=path, split_info=split_info, skew=0.9)
'''
def create_shortcut(spurious_corr, df=None, path=None, split_info=None, skew=1.0):
    
    if path is not None:
        df = pd.read_csv(path)
    
    if split_info is not None:
        col = split_info['col_name']
        values = split_info['values']
        arr = [False]*len(df)
        for val in values:
            arr = ( arr | (df[col]==val) )
        df = df[arr]
        
    bool_arr = {}
    bool_arr['core'] = []
    bool_arr['spurious'] = []
    for key in spurious_corr:
        if len(spurious_corr[key])==2:
            flag = 'range' # it's a continuous variable, apply <,>=
        elif len(spurious_corr[key])==3:
            flag = 'hard' # it's a discrete assignment, apply =
        else:
            raise('Invalid "spurious_corr" dictionary')
            
        if flag=='range':
            attr = spurious_corr[key][0]
            thresh = spurious_corr[key][1]
            
            arr_temp = (df[attr]<thresh)
            bool_arr[key].append(arr_temp)
            
            arr_temp = (df[attr]>=thresh)
            bool_arr[key].append(arr_temp)
        elif flag=='hard':
            attr = spurious_corr[key][0]
            thresh1 = spurious_corr[key][1]
            thresh2 = spurious_corr[key][2]
            
            arr_temp = (df[attr]==thresh1)
            bool_arr[key].append(arr_temp)
            
            arr_temp = (df[attr]==thresh2)
            bool_arr[key].append(arr_temp)
        else:
            raise('Invalid flag variable')
            
    # all diagonal elements must be present in most cases
    arr00 = ((bool_arr['core'][0])&(bool_arr['spurious'][0]))
    arr11 = ((bool_arr['core'][1])&(bool_arr['spurious'][1]))
    arr01 = ((bool_arr['core'][0])&(bool_arr['spurious'][1]))
    arr10 = ((bool_arr['core'][1])&(bool_arr['spurious'][0]))
    
    df00 = df[arr00]
    df11 = df[arr11]
    df01 = df[arr01]
    df10 = df[arr10]
    df_diag = pd.concat([df00,df11]).sample(frac=1)
    df_nondiag = pd.DataFrame()
    if skew == 1.0:
        df_final = df_diag
    else:
        coeff = (1/skew-1)
        num_nondiag = int(len(df_diag)*coeff) # number of non-diag samples needed
                                                # to satisfy skew requirement
        df_nondiag = pd.concat([df10,df01]).sample(frac=1)
        if len(df_nondiag)>=num_nondiag: # meaning you have required num of samples
            df_nondiag = df_nondiag.sample(n=num_nondiag)
        else:
            # reduce number of diag samples to fulfill ratio
            num_diag = int(len(df_nondiag)/coeff)
            df_diag = df_diag.sample(n=num_diag)
            
        df_final = pd.concat([df_diag,df_nondiag]).sample(frac=1)    
        
    # get group stats
    bool_arr = {}
    bool_arr['core'] = []
    bool_arr['spurious'] = []
    for key in spurious_corr:
        if len(spurious_corr[key])==2:
            flag = 'range' # it's a continuous variable, apply <,>=
        elif len(spurious_corr[key])==3:
            flag = 'hard' # it's a discrete assignment, apply =
        else:
            raise('Invalid "spurious_corr" dictionary')
            
        if flag=='range':
            attr = spurious_corr[key][0]
            thresh = spurious_corr[key][1]
            
            arr_temp = (df_final[attr]<thresh)
            bool_arr[key].append(arr_temp)
            
            arr_temp = (df_final[attr]>=thresh)
            bool_arr[key].append(arr_temp)
        elif flag=='hard':
            attr = spurious_corr[key][0]
            thresh1 = spurious_corr[key][1]
            thresh2 = spurious_corr[key][2]
            
            arr_temp = (df_final[attr]==thresh1)
            bool_arr[key].append(arr_temp)
            
            arr_temp = (df_final[attr]==thresh2)
            bool_arr[key].append(arr_temp)
        else:
            raise('Invalid flag variable')

    arr00 = ((bool_arr['core'][0])&(bool_arr['spurious'][0]))
    arr11 = ((bool_arr['core'][1])&(bool_arr['spurious'][1]))
    arr01 = ((bool_arr['core'][0])&(bool_arr['spurious'][1]))
    arr10 = ((bool_arr['core'][1])&(bool_arr['spurious'][0]))
    
    df00 = df_final[arr00]
    df11 = df_final[arr11]
    df01 = df_final[arr01]
    df10 = df_final[arr10]
    
    group_dfs = {'df_c0_s0':df00,'df_c0_s1':df01,'df_c1_s0':df10,'df_c1_s1':df11}

    print('core: ')
    print(spurious_corr['core'])
    print('\n')
    print('spurious: ')
    print(spurious_corr['spurious'])
    print('\n')
    print('Groups: ')
    print('len(df_c0_s0): %d;          len(df_c0_s1): %d' %(len(df00),len(df01)))
    print('len(df_c1_s0): %d;          len(df_c1_s1): %d' %(len(df10),len(df11)))

    return df_final, group_dfs


'''
Usage:
spurious_corr = {'core':['Pneumothorax',0.0,1.0], 'spurious':['view','PA','AP']}
groups = retrieve_groups(spurious_corr=spurious_corr, path=path)
'''
def retrieve_groups(spurious_corr, df=None, path=None):
    
    if path is not None:
        df = pd.read_csv(path)    
        
    # get group stats
    bool_arr = {}
    bool_arr['core'] = []
    bool_arr['spurious'] = []
    for key in spurious_corr:
        if len(spurious_corr[key])==2:
            flag = 'range' # it's a continuous variable, apply <,>=
        elif len(spurious_corr[key])==3:
            flag = 'hard' # it's a discrete assignment, apply =
        else:
            raise('Invalid "spurious_corr" dictionary')
            
        if flag=='range':
            attr = spurious_corr[key][0]
            thresh = spurious_corr[key][1]
            
            arr_temp = (df[attr]<thresh)
            bool_arr[key].append(arr_temp)
            
            arr_temp = (df[attr]>=thresh)
            bool_arr[key].append(arr_temp)
        elif flag=='hard':
            attr = spurious_corr[key][0]
            thresh1 = spurious_corr[key][1]
            thresh2 = spurious_corr[key][2]
            
            arr_temp = (df[attr]==thresh1)
            bool_arr[key].append(arr_temp)
            
            arr_temp = (df[attr]==thresh2)
            bool_arr[key].append(arr_temp)
        else:
            raise('Invalid flag variable')

    arr00 = ((bool_arr['core'][0])&(bool_arr['spurious'][0]))
    arr11 = ((bool_arr['core'][1])&(bool_arr['spurious'][1]))
    arr01 = ((bool_arr['core'][0])&(bool_arr['spurious'][1]))
    arr10 = ((bool_arr['core'][1])&(bool_arr['spurious'][0]))
    
    df00 = df[arr00]
    df11 = df[arr11]
    df01 = df[arr01]
    df10 = df[arr10]
    
    group_dfs = {'df_c0_s0':df00,'df_c0_s1':df01,'df_c1_s0':df10,'df_c1_s1':df11}

    print('core: ')
    print(spurious_corr['core'])
    print('\n')
    print('spurious: ')
    print(spurious_corr['spurious'])
    print('\n')
    print('Groups: ')
    print('len(df_c0_s0): %d;          len(df_c0_s1): %d' %(len(df00),len(df01)))
    print('len(df_c1_s0): %d;          len(df_c1_s1): %d' %(len(df10),len(df11)))

    return group_dfs


# Store -> get_df -> get_metrics form one pipeline
# as you iterate through data loader, store your values
# use get_df to convert that into a suitable outcomes df
# feed this df into get_metrics to see all metrics on your classifier
    
# use this to store (concatenate) your predictions or labels or ... during training
# it maintains a list of variables (lov): can be cuda tensors or lists
# keep feeding it list of one dimensional tensors (squeezing automatically done), or lists
# and it'll keep concatenating them to existing list
class Store():
    
    def __init__(self):
        self.lov = [] 
        self.start_flag = True
        
    def feed(self, mini_lov):
        
        if self.start_flag:
            self.start_flag = False

            for var in mini_lov:
                if isinstance(var, list):
                    self.lov.append(var)
                elif torch.is_tensor(var):
                    if len(var.shape)>1:
                        var = var.squeeze()
                    self.lov.append(var)
                else:
                    raise ValueError('Invalid input type for class "Store"!')
                    
        else:
            for i in range(len(mini_lov)):
                if isinstance(mini_lov[i], list):
                    self.lov[i] += mini_lov[i]

                elif torch.is_tensor(mini_lov[i]):
                    tensor1 = mini_lov[i]            
                    if len(tensor1.shape)>1:
                        tensor1 = tensor1.squeeze()
                    
                    tensor2 = self.lov[i]
                    self.lov[i] = torch.cat((tensor2,tensor1))

                else:
                    raise ValueError('Invalid input type for class "Store"!')
    
# pass the Stores lov into this to convert it into suitable df
# that can be passed to get_metrics
def get_df(lov, cols):
    
    df = pd.DataFrame()
    
    for idx, col in enumerate(cols):
        if isinstance(lov[idx],list):
            df[col] = lov[idx]
        elif torch.is_tensor(lov[idx]):
            df[col] = lov[idx].detach().cpu()
        else:
            raise ValueError('Invalid input type for "get_df"!')
        
    return df

# push list of tensors CPU
def to_cpu(arr):
    for idx,x in enumerate(arr):
        arr[idx] = x.to('cpu')
    return arr

# print GPU memory profile
def print_memory_profile(s):
    # print GPU memory
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print(s)
    print(t/1024**3,r/1024**3,a/1024**3)
    print('\n')