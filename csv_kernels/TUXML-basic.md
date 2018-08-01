

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TUXML_CSV_FILENAME="./config_bdd.csv"

# sanity check CSV
with open(TUXML_CSV_FILENAME, "r") as file:
    k = file.readline()
    t = k.split(",")
    s = set(t)
    assert(len(t) == len(s)) # unique number of options/features/column names

# parsing for real with pandas 
rawtuxdata = pd.read_csv(open(TUXML_CSV_FILENAME, "r"))

basic_head = ["cid", "time", "date"] # "compile"
size_methods = ["vmlinux", "GZIP-bzImage", "GZIP-vmlinux", "GZIP", "BZIP2-bzImage", 
              "BZIP2-vmlinux", "BZIP2", "LZMA-bzImage", "LZMA-vmlinux", "LZMA", "XZ-bzImage", "XZ-vmlinux", "XZ", 
              "LZO-bzImage", "LZO-vmlinux", "LZO", "LZ4-bzImage", "LZ4-vmlinux", "LZ4"]


### basic stats about options and remove of unique values 
## could be improved 

tri_state_values = ['y', 'n', 'm']

ftuniques = []
freq_ymn_features = []
non_tristate_options = []

for col in rawtuxdata:
    ft = rawtuxdata[col]    
    # eg always "y"
    if len(ft.unique()) == 1:
        ftuniques.append(col)
    # only tri-state values (y, n, m) (possible TODO: handle numerical/string options)    
    elif all(x in tri_state_values for x in ft.unique()):     #len(ft.unique()) == 3: 
        freq = ft.value_counts(normalize=True)
        freqy = 0
        freqn = 0
        freqm = 0
        if ('y' in freq.index):
            freqy = freq['y']
        if ('n' in freq.index):
            freqn = freq['n']
        if ('m' in freq.index):
            freqm = freq['m']
        freq_ymn_features.append((col, freqy, freqm, freqn))
    else:
        if not (col in size_methods): 
            non_tristate_options.append(col)
        

### TODO: we want to keep all quantitative values!
# non_tristate_options.remove('LZO') # ('vmlinux')

# we want to keep measurements (that are not tristate ;)) 
# non_tristate_options = list(set(non_tristate_options) - set(size_methods))

#### print options with unique values
# options with only one value eg always "y"
#i = 0
#for ft in ftuniques:
#    print(ft + " (" + str(i) + ")")
#    i = i + 1

print("Original size (#configs/#options) of the dataset " + str(rawtuxdata.shape))
print ("Number of options with only one value (eg always y): " + str(pd.DataFrame(ftuniques).shape))

# maybe we can drop options with only one unique value (no interest for machine learning)
# TODO: maybe we can rely on more traditional feature reduction techniques
# TODO: need to think about *when* to apply the removal 
rawtuxdata.drop(columns=ftuniques,inplace=True) 
## non_tristate_options include basic stuff like date, time, cid but also string/numerical options
print ("Non tri-state value options (eg string or integer or hybrid values): " 
       + str(pd.DataFrame(non_tristate_options).shape) + " ") 
#      + str(pd.DataFrame(non_tristate_options)))


print ("Predictor variables: " + str(rawtuxdata.drop(columns=non_tristate_options).columns.size))
# frequency of y, m, and n values 
#plt.figure()
#pd.DataFrame(freq_ymn_features, columns=["feature", "freqy", "freqm", "freqn"]).plot(kind='hist', alpha=0.8) #plot()
#plt.show()


    


```

    /usr/lib/python3/dist-packages/IPython/core/interactiveshell.py:2718: DtypeWarning: Columns (1150,2722,6015,6026,6717,7350,7676,7726,10442) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)


    Original size (#configs/#options) of the dataset (2500, 12798)
    Number of options with only one value (eg always y): (3424, 1)
    Non tri-state value options (eg string or integer or hybrid values): (144, 1) 
    Predictor variables: 9230



```python
'X86_64' in ftuniques, 'DEBUG_INFO' in ftuniques, 'GCOV_PROFILE_ALL' in ftuniques, 'KASAN' in ftuniques, 'UBSAN_SANITIZE_ALL' in ftuniques, 'RELOCATABLE' in ftuniques, 'XFS_DEBUG' in ftuniques, 'AIC7XXX_BUILD_FIRMWARE' in ftuniques, 'AIC79XX_BUILD_FIRMWARE' in ftuniques, 'WANXL_BUILD_FIRMWARE' in ftuniques
```




    (True, True, True, True, True, False, True, True, True, True)




```python
if 'RELOCATABLE' in rawtuxdata.columns:
    print(rawtuxdata.query("RELOCATABLE == 'y'")[['cid', 'RELOCATABLE']])
```

            cid RELOCATABLE
    6     90006           y
    10    90010           y
    21    90021           y
    22    90022           y
    27    90027           y
    39    90039           y
    52    90052           y
    69    90069           y
    71    90071           y
    86    90086           y
    98    90098           y
    107   90107           y
    109   90109           y
    120   90120           y
    125   90125           y
    132   90132           y
    134   90134           y
    138   90138           y
    153   90153           y
    168   90168           y
    175   90175           y
    181   90181           y
    185   90185           y
    192   90192           y
    195   90195           y
    215   90215           y
    240   90240           y
    241   90241           y
    248   90248           y
    251   90251           y
    ...     ...         ...
    2229  92229           y
    2237  92237           y
    2241  92241           y
    2251  92251           y
    2253  92253           y
    2255  92255           y
    2276  92276           y
    2277  92277           y
    2283  92283           y
    2290  92290           y
    2291  92291           y
    2293  92293           y
    2303  92303           y
    2353  92353           y
    2357  92357           y
    2398  92398           y
    2402  92402           y
    2405  92405           y
    2419  92419           y
    2421  92421           y
    2434  92434           y
    2439  92439           y
    2445  92445           y
    2455  92455           y
    2461  92461           y
    2465  92465           y
    2468  92468           y
    2473  92473           y
    2478  92478           y
    2491  92491           y
    
    [268 rows x 2 columns]



```python
print("Data exploration")
```

    Data exploration



```python
# BUGS EXPLORATION
def bug_exploration():
    rawtuxdata.query("AIC7XXX_BUILD_FIRMWARE == 'y'")[['cid', 'vmlinux']]
    rawtuxdata.query("AIC79XX_BUILD_FIRMWARE == 'y'")[['cid', 'vmlinux']]
    rawtuxdata.query("WANXL_BUILD_FIRMWARE == 'y'")[['cid', 'vmlinux']]
    rawtuxdata.query("GENERIC_ALLOCATOR == 'n' & DRM_VBOXVIDEO == 'y'")[['cid', 'vmlinux']]
    rawtuxdata.query("GENERIC_ALLOCATOR == 'y' & DRM_VBOXVIDEO == 'y'")[['cid', 'vmlinux']]
    rawtuxdata.query("GENERIC_ALLOCATOR == 'n' & DRM_VBOXVIDEO == 'm'")[['cid', 'vmlinux']]
    return rawtuxdata.query("DRM_VBOXVIDEO == 'y'")[['cid', 'vmlinux']]

# bug_exploration()




```


```python
#rawtuxdata[rawtuxdata['X86_64'] == 'n']
#rawtuxdata.query("X86_64 == 'n'")
```


```python
#rawtuxdata[(rawtuxdata['DEBUG_INFO'] == 'n') & (rawtuxdata['GCOV_PROFILE_ALL'] == 'n') & (rawtuxdata['KASAN'] == 'n') & (rawtuxdata['MODULES'] == 'y')]
# rawtuxdata.query("(DEBUG_INFO == 'n') & (GCOV_PROFILE_ALL == 'n') & (KASAN == 'n') & (MODULES == 'y')")
#rawtuxdata.query("(DEBUG_INFO == 'n') & (GCOV_PROFILE_ALL == 'n') & (KASAN == 'n')").shape, rawtuxdata.shape

```


```python
#rawtuxdata[rawtuxdata['vmlinux'] == 1168072][['cid', 'CC_OPTIMIZE_FOR_SIZE', 'DEBUG_INFO_DWARF4', 'KASAN', 'UBSAN_ALIGNMENT', 'X86_NEED_RELOCS', 'RANDOMIZE_BASE', 'GCOV_PROFILE_ALL', 'UBSAN_SANITIZE_ALL', 'DEBUG_INFO', 'MODULES', 'DEBUG_INFO_REDUCED', 'DEBUG_INFO_SPLIT']]
tiny_data = rawtuxdata.query("vmlinux == 1168072") #tiny config for X86_32
#if (len(tiny_data) > 0):
#    print(tiny_data[['cid', 'CC_OPTIMIZE_FOR_SIZE', 'DEBUG_INFO_DWARF4', 'KASAN', 'UBSAN_ALIGNMENT', 'X86_NEED_RELOCS', 'RANDOMIZE_BASE', 'GCOV_PROFILE_ALL', 'UBSAN_SANITIZE_ALL', 'DEBUG_INFO', 'MODULES', 'DEBUG_INFO_REDUCED', 'DEBUG_INFO_SPLIT']])
```


```python
#rawtuxdata[rawtuxdata['vmlinux'] == -1]
rawtuxdata.query("vmlinux == -1")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cid</th>
      <th>date</th>
      <th>time</th>
      <th>vmlinux</th>
      <th>GZIP-bzImage</th>
      <th>GZIP-vmlinux</th>
      <th>GZIP</th>
      <th>BZIP2-bzImage</th>
      <th>BZIP2-vmlinux</th>
      <th>BZIP2</th>
      <th>...</th>
      <th>VIDEO_S3C_CAMIF</th>
      <th>SND_SOC_INTEL_SKL_NAU88L25_SSM4567_MACH</th>
      <th>APDS9960</th>
      <th>SLABINFO</th>
      <th>MICROCODE_AMD</th>
      <th>ISDN_DRV_HISAX</th>
      <th>CHARGER_BQ24190</th>
      <th>SND_SOC_NAU8825</th>
      <th>BH1750</th>
      <th>NETWORK_FILESYSTEMS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>506</th>
      <td>90506</td>
      <td>2018-07-31 13:28:32</td>
      <td>397.998</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>710</th>
      <td>90710</td>
      <td>2018-07-31 14:55:26</td>
      <td>762.980</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>975</th>
      <td>90975</td>
      <td>2018-07-31 16:34:17</td>
      <td>557.827</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>1054</th>
      <td>91054</td>
      <td>2018-07-31 17:00:55</td>
      <td>563.142</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>1189</th>
      <td>91189</td>
      <td>2018-07-31 17:36:52</td>
      <td>614.056</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>1291</th>
      <td>91291</td>
      <td>2018-07-31 18:05:21</td>
      <td>2063.690</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
    </tr>
    <tr>
      <th>1633</th>
      <td>91633</td>
      <td>2018-07-31 19:33:00</td>
      <td>647.727</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>1742</th>
      <td>91742</td>
      <td>2018-07-31 19:58:26</td>
      <td>581.922</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>1862</th>
      <td>91862</td>
      <td>2018-07-31 20:29:41</td>
      <td>1787.870</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>m</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>2072</th>
      <td>92072</td>
      <td>2018-07-31 21:22:43</td>
      <td>673.648</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>2120</th>
      <td>92120</td>
      <td>2018-07-31 21:37:08</td>
      <td>701.833</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 9374 columns</p>
</div>




```python
#rawtuxdata[rawtuxdata['vmlinux'] == 1168072]['MODULES']
rawtuxdata.query("vmlinux == 1168072")['MODULES'] #tiny config for X86_32
```




    Series([], Name: MODULES, dtype: object)




```python
# playing a bit with the data 
rawtuxdata.dtypes
# 'DEBUG_INFOO' in list(pd.DataFrame(non_tristate_options)[0]) # 
# tuxdata['DEBUG_INFO'].unique()
#tuxdata['OUTPUT_FORMAT'].dtypes
#tuxdata['DEFAULT_HOSTNAME'].unique()

#rawtuxdata[:5]
rawtuxdata[:20]['vmlinux']
#tuxdata[:5]['CONFIG_DEBUG_INFO']
#tuxdata['ARCH_HAS_SG_CHAIN'].unique()
#tuxdata['ARCH_HAS_SG_CHAIN'].astype('category')
```




    0     21865256
    1     42817248
    2     21952696
    3     20877312
    4     17423400
    5     22586312
    6     55507888
    7     39160712
    8     25672992
    9     21195640
    10    21579408
    11    26888032
    12    25378496
    13    18594192
    14    32391512
    15    32438224
    16    22446600
    17    32961592
    18    18669624
    19    38557400
    Name: vmlinux, dtype: int64




```python
rawtuxdata.shape, rawtuxdata.query("vmlinux != -1").shape
```




    ((2500, 9374), (2489, 9374))




```python
print("some configurations may have X86_32 (coz we have tested/tried some options and there are in the database)")
# we only keep X86_64 configurations
#rawtuxdata = rawtuxdata[rawtuxdata['X86_64'] == 'y'] ### TODO: I've impression it's not the most effective way (wrt memory) to filter 
if 'X86_64' in rawtuxdata.columns:
    print(rawtuxdata['X86_64'].describe())
    rawtuxdata.query("X86_64 == 'y'", inplace=True)
rawtuxdata.info(memory_usage='deep')
```

    some configurations may have X86_32 (coz we have tested/tried some options and there are in the database)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2500 entries, 0 to 2499
    Columns: 9374 entries, cid to NETWORK_FILESYSTEMS
    dtypes: float64(1), int64(133), object(9240)
    memory usage: 1.4 GB



```python
from sklearn.preprocessing import *
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn import tree
import graphviz 


LEARN_COMPILATION_SUCCESS = False # costly in time and space 
compilation_status_column_name = 'compile_success'

def encode_data_compilation(rawtuxdata):
    lae = LabelEncoder()
    # we save quantitative values we want (here vmlinux, TODO: generalize)
    # the key idea is that the labelling encoder should not be applied to this kind of values (only to predictor variables!)
    # vml = rawtuxdata['LZO'] # rawtuxdata['vmlinux'] 
    o_sizes = rawtuxdata[size_methods]

    # we remove non tri state options, but TODO there are perhaps some interesting options (numerical or string) here
    #tuxdata = rawtuxdata.drop(columns=non_tristate_options).drop(columns=['vmlinux']).apply(le.fit_transform)
    tuxdata_for_compilation = rawtuxdata.drop(columns=non_tristate_options).drop(columns=size_methods).apply(lae.fit_transform)

    #tuxdata['vmlinux'] = vml 
    tuxdata_for_compilation[size_methods] = o_sizes
    # we can ue vmlinux since it has been restored thanks to previous line
    tuxdata_for_compilation[compilation_status_column_name] = tuxdata_for_compilation['vmlinux'] != -1
    return tuxdata_for_compilation

def learn_compilation_success(tuxdata_for_compilation):
    TESTING_SIZE=0.3 
    X_train, X_test, y_train, y_test = train_test_split(tuxdata_for_compilation.drop(columns=size_methods).drop(columns=compilation_status_column_name), tuxdata_for_compilation[compilation_status_column_name], test_size=TESTING_SIZE, random_state=0)  
    clf = tree.DecisionTreeClassifier() #GradientBoostingClassifier(n_estimators=100) #RandomForestRegressor(n_estimators=100) #   #GradientBoostingRegressor(n_estimators=100)  
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]    

    TOP_FT_IMPORTANCE=20
    print("Feature ranking: " + "top (" + str(TOP_FT_IMPORTANCE) + ")")
    for f in range(TOP_FT_IMPORTANCE): # len(indices)
        print("%d. feature %s %d (%f)" % (f + 1, tuxdata_for_compilation.columns[indices[f]], indices[f], importances[indices[f]]))
   
    
    dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=tuxdata_for_compilation.drop(columns=size_methods).drop(columns=compilation_status_column_name).columns,  
                         filled=True, rounded=True,
                         special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph.render("TUXML compilation sucess")
    
    acc = accuracy_score (y_test, y_pred)
    prec = precision_score (y_test, y_pred)
    reca = recall_score (y_test, y_pred)
    f1 = f1_score (y_test, y_pred)
    print("Accuracy score: %.2f" % (acc))
    print("Precision score: %.2f" % (prec))
    print("Recall score: %.2f" % (reca))
    print("F1 score: %.2f" % (f1))

if (LEARN_COMPILATION_SUCCESS):
    tuxdata_for_compilation = encode_data_compilation(rawtuxdata)
    tuxdata_for_compilation [compilation_status_column_name].describe()
    learn_compilation_success(tuxdata_for_compilation)
```


```python
#rawtuxdata.query("vmlinux == -1")[['cid', 'AIC7XXX_BUILD_FIRMWARE', 'AIC79XX_BUILD_FIRMWARE', 'IPVTAP', 'WANXL_BUILD_FIRMWARE', 'TCIC']]
```


```python
# aka MAPE
def mean_relative_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

```


```python
# remove entries with same configurations
print(str(len(rawtuxdata)) + " before the removal of some entries (those with same configurations)")
# tuxdata.drop_duplicates(subset=tuxdata.columns.difference(['vmlinux']), inplace=True)
rawtuxdata.drop_duplicates(subset=rawtuxdata.columns.difference(size_methods).difference(basic_head), inplace=True)
print(str(len(rawtuxdata)) + " after the removal of some entries (those with same configurations)")

#n_failures = len(tuxdata[~np.isnan(tuxdata['vmlinux'])])
#n_failures = len(rawtuxdata.query("vmlinux != -1")) #len(tuxdata[np.isnan(tuxdata['vmlinux'])])
#print(str(n_failures) + " non-failures out of " + str(len(rawtuxdata)))

#tuxdata = tuxdata[~np.isnan(tuxdata['vmlinux'])]
#rawtuxdata = rawtuxdata[rawtuxdata['vmlinux'] != -1] #tuxdata[~np.isnan(tuxdata['vmlinux'])]
rawtuxdata.query("(vmlinux != -1) & (vmlinux != 0)", inplace=True)
print(str(len(rawtuxdata)) + " after the removal of configurations that do NOT compile")

```

    2500 before the removal of some entries (those with same configurations)
    2500 after the removal of some entries (those with same configurations)
    2489 after the removal of configurations that do NOT compile



```python
rawtuxdata.query("vmlinux == 1168072") # tinyconfig with X86_32
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cid</th>
      <th>date</th>
      <th>time</th>
      <th>vmlinux</th>
      <th>GZIP-bzImage</th>
      <th>GZIP-vmlinux</th>
      <th>GZIP</th>
      <th>BZIP2-bzImage</th>
      <th>BZIP2-vmlinux</th>
      <th>BZIP2</th>
      <th>...</th>
      <th>VIDEO_S3C_CAMIF</th>
      <th>SND_SOC_INTEL_SKL_NAU88L25_SSM4567_MACH</th>
      <th>APDS9960</th>
      <th>SLABINFO</th>
      <th>MICROCODE_AMD</th>
      <th>ISDN_DRV_HISAX</th>
      <th>CHARGER_BQ24190</th>
      <th>SND_SOC_NAU8825</th>
      <th>BH1750</th>
      <th>NETWORK_FILESYSTEMS</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 9374 columns</p>
</div>




```python
rawtuxdata.query("vmlinux == 7317008") # tiny config for X86_64
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cid</th>
      <th>date</th>
      <th>time</th>
      <th>vmlinux</th>
      <th>GZIP-bzImage</th>
      <th>GZIP-vmlinux</th>
      <th>GZIP</th>
      <th>BZIP2-bzImage</th>
      <th>BZIP2-vmlinux</th>
      <th>BZIP2</th>
      <th>...</th>
      <th>VIDEO_S3C_CAMIF</th>
      <th>SND_SOC_INTEL_SKL_NAU88L25_SSM4567_MACH</th>
      <th>APDS9960</th>
      <th>SLABINFO</th>
      <th>MICROCODE_AMD</th>
      <th>ISDN_DRV_HISAX</th>
      <th>CHARGER_BQ24190</th>
      <th>SND_SOC_NAU8825</th>
      <th>BH1750</th>
      <th>NETWORK_FILESYSTEMS</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 9374 columns</p>
</div>




```python


```


```python
plt.figure()
pd.DataFrame(rawtuxdata['vmlinux']).plot.box()
plt.show(block=False)

plt.figure()
pd.DataFrame(rawtuxdata['LZO']).plot.box()
plt.show(block=False)

plt.figure()
pd.DataFrame(rawtuxdata['BZIP2']).plot.box()
plt.show(block=False)


rawtuxdata['vmlinux'].describe()

```


    <Figure size 432x288 with 0 Axes>



![png](TUXML-basic_files/TUXML-basic_20_1.png)



    <Figure size 432x288 with 0 Axes>



![png](TUXML-basic_files/TUXML-basic_20_3.png)



    <Figure size 432x288 with 0 Axes>



![png](TUXML-basic_files/TUXML-basic_20_5.png)





    count    2.489000e+03
    mean     3.064421e+07
    std      1.224472e+07
    min      1.088954e+07
    25%      2.186526e+07
    50%      2.775274e+07
    75%      3.647359e+07
    max      1.298963e+08
    Name: vmlinux, dtype: float64




```python
rawtuxdata.query("vmlinux == 1168072") # tiny config for X86_32
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cid</th>
      <th>date</th>
      <th>time</th>
      <th>vmlinux</th>
      <th>GZIP-bzImage</th>
      <th>GZIP-vmlinux</th>
      <th>GZIP</th>
      <th>BZIP2-bzImage</th>
      <th>BZIP2-vmlinux</th>
      <th>BZIP2</th>
      <th>...</th>
      <th>VIDEO_S3C_CAMIF</th>
      <th>SND_SOC_INTEL_SKL_NAU88L25_SSM4567_MACH</th>
      <th>APDS9960</th>
      <th>SLABINFO</th>
      <th>MICROCODE_AMD</th>
      <th>ISDN_DRV_HISAX</th>
      <th>CHARGER_BQ24190</th>
      <th>SND_SOC_NAU8825</th>
      <th>BH1750</th>
      <th>NETWORK_FILESYSTEMS</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 9374 columns</p>
</div>




```python
import scipy.stats
import seaborn as sns



def color_negative_positive(val, pcolor="green", ncolor="red"):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = pcolor if val > 0 else ncolor 
    if val == 0:
        color = 'black' 
    return 'color: %s' % color

compress_methods = ["GZIP", "BZIP2", "LZMA", "XZ", "LZO", "LZ4"]
def compareCompress(size_measure_of_interest): #"" # "-vmlinux" #"-bzImage" # prefix
    rCompressDiff = pd.DataFrame(index=list(map(lambda c: c + "o", compress_methods)) , columns=compress_methods) 
    for compress_method in compress_methods:
        for compress_method2 in compress_methods:
            rCompressDiff.loc[compress_method + "o"][compress_method2] = (np.mean(rawtuxdata[compress_method + size_measure_of_interest] / rawtuxdata[compress_method2 + size_measure_of_interest]) * 100) - 100
    return rCompressDiff

#cmy = sns.light_palette("red", as_cmap=True)
compareCompress("").style.set_caption('Difference (average in percentage) per compression methods').applymap(color_negative_positive)
```




<style  type="text/css" >
    #T_848b4a08_9564_11e8_a1ed_525400123456row0_col0 {
            color:  black;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row0_col1 {
            color:  green;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row0_col2 {
            color:  green;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row0_col3 {
            color:  green;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row0_col4 {
            color:  red;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row0_col5 {
            color:  red;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row1_col0 {
            color:  red;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row1_col1 {
            color:  black;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row1_col2 {
            color:  green;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row1_col3 {
            color:  green;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row1_col4 {
            color:  red;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row1_col5 {
            color:  red;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row2_col0 {
            color:  red;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row2_col1 {
            color:  red;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row2_col2 {
            color:  black;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row2_col3 {
            color:  green;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row2_col4 {
            color:  red;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row2_col5 {
            color:  red;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row3_col0 {
            color:  red;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row3_col1 {
            color:  red;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row3_col2 {
            color:  red;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row3_col3 {
            color:  black;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row3_col4 {
            color:  red;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row3_col5 {
            color:  red;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row4_col0 {
            color:  green;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row4_col1 {
            color:  green;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row4_col2 {
            color:  green;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row4_col3 {
            color:  green;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row4_col4 {
            color:  black;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row4_col5 {
            color:  red;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row5_col0 {
            color:  green;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row5_col1 {
            color:  green;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row5_col2 {
            color:  green;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row5_col3 {
            color:  green;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row5_col4 {
            color:  green;
        }    #T_848b4a08_9564_11e8_a1ed_525400123456row5_col5 {
            color:  black;
        }</style>  
<table id="T_848b4a08_9564_11e8_a1ed_525400123456" ><caption>Difference (average in percentage) per compression methods</caption> 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >GZIP</th> 
        <th class="col_heading level0 col1" >BZIP2</th> 
        <th class="col_heading level0 col2" >LZMA</th> 
        <th class="col_heading level0 col3" >XZ</th> 
        <th class="col_heading level0 col4" >LZO</th> 
        <th class="col_heading level0 col5" >LZ4</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_848b4a08_9564_11e8_a1ed_525400123456level0_row0" class="row_heading level0 row0" >GZIPo</th> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row0_col0" class="data row0 col0" >0</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row0_col1" class="data row0 col1" >3.20229</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row0_col2" class="data row0 col2" >22.6095</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row0_col3" class="data row0 col3" >35.1338</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row0_col4" class="data row0 col4" >-9.11279</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row0_col5" class="data row0 col5" >-14.6803</td> 
    </tr>    <tr> 
        <th id="T_848b4a08_9564_11e8_a1ed_525400123456level0_row1" class="row_heading level0 row1" >BZIP2o</th> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row1_col0" class="data row1 col0" >-3.08904</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row1_col1" class="data row1 col1" >0</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row1_col2" class="data row1 col2" >18.8241</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row1_col3" class="data row1 col3" >30.972</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row1_col4" class="data row1 col4" >-11.9275</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row1_col5" class="data row1 col5" >-17.3265</td> 
    </tr>    <tr> 
        <th id="T_848b4a08_9564_11e8_a1ed_525400123456level0_row2" class="row_heading level0 row2" >LZMAo</th> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row2_col0" class="data row2 col0" >-18.4303</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row2_col1" class="data row2 col1" >-15.8168</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row2_col2" class="data row2 col2" >0</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row2_col3" class="data row2 col3" >10.2225</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row2_col4" class="data row2 col4" >-25.8597</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row2_col5" class="data row2 col5" >-30.3996</td> 
    </tr>    <tr> 
        <th id="T_848b4a08_9564_11e8_a1ed_525400123456level0_row3" class="row_heading level0 row3" >XZo</th> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row3_col0" class="data row3 col0" >-25.7453</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row3_col1" class="data row3 col1" >-23.3605</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row3_col2" class="data row3 col2" >-8.96217</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row3_col3" class="data row3 col3" >0</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row3_col4" class="data row3 col4" >-32.5141</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row3_col5" class="data row3 col5" >-36.6429</td> 
    </tr>    <tr> 
        <th id="T_848b4a08_9564_11e8_a1ed_525400123456level0_row4" class="row_heading level0 row4" >LZOo</th> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row4_col0" class="data row4 col0" >10.0362</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row4_col1" class="data row4 col1" >13.5506</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row4_col2" class="data row4 col2" >34.9219</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row4_col3" class="data row4 col3" >48.6904</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row4_col4" class="data row4 col4" >0</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row4_col5" class="data row4 col5" >-6.12937</td> 
    </tr>    <tr> 
        <th id="T_848b4a08_9564_11e8_a1ed_525400123456level0_row5" class="row_heading level0 row5" >LZ4o</th> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row5_col0" class="data row5 col0" >17.2289</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row5_col1" class="data row5 col1" >20.9672</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row5_col2" class="data row5 col2" >43.7451</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row5_col3" class="data row5 col3" >58.4235</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row5_col4" class="data row5 col4" >6.53242</td> 
        <td id="T_848b4a08_9564_11e8_a1ed_525400123456row5_col5" class="data row5 col5" >0</td> 
    </tr></tbody> 
</table> 




```python
compareCompress("-bzImage").style.set_caption('Difference (average in percentage) per compression methods, bzImage').applymap(color_negative_positive)

```




<style  type="text/css" >
    #T_848b4a09_9564_11e8_a1ed_525400123456row0_col0 {
            color:  black;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row0_col1 {
            color:  red;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row0_col2 {
            color:  green;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row0_col3 {
            color:  green;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row0_col4 {
            color:  red;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row0_col5 {
            color:  red;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row1_col0 {
            color:  green;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row1_col1 {
            color:  black;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row1_col2 {
            color:  green;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row1_col3 {
            color:  green;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row1_col4 {
            color:  green;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row1_col5 {
            color:  green;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row2_col0 {
            color:  red;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row2_col1 {
            color:  red;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row2_col2 {
            color:  black;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row2_col3 {
            color:  green;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row2_col4 {
            color:  red;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row2_col5 {
            color:  red;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row3_col0 {
            color:  red;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row3_col1 {
            color:  red;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row3_col2 {
            color:  red;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row3_col3 {
            color:  black;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row3_col4 {
            color:  red;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row3_col5 {
            color:  red;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row4_col0 {
            color:  green;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row4_col1 {
            color:  red;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row4_col2 {
            color:  green;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row4_col3 {
            color:  green;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row4_col4 {
            color:  black;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row4_col5 {
            color:  red;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row5_col0 {
            color:  green;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row5_col1 {
            color:  red;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row5_col2 {
            color:  green;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row5_col3 {
            color:  green;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row5_col4 {
            color:  green;
        }    #T_848b4a09_9564_11e8_a1ed_525400123456row5_col5 {
            color:  black;
        }</style>  
<table id="T_848b4a09_9564_11e8_a1ed_525400123456" ><caption>Difference (average in percentage) per compression methods, bzImage</caption> 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >GZIP</th> 
        <th class="col_heading level0 col1" >BZIP2</th> 
        <th class="col_heading level0 col2" >LZMA</th> 
        <th class="col_heading level0 col3" >XZ</th> 
        <th class="col_heading level0 col4" >LZO</th> 
        <th class="col_heading level0 col5" >LZ4</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_848b4a09_9564_11e8_a1ed_525400123456level0_row0" class="row_heading level0 row0" >GZIPo</th> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row0_col0" class="data row0 col0" >0</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row0_col1" class="data row0 col1" >-34.8382</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row0_col2" class="data row0 col2" >22.2373</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row0_col3" class="data row0 col3" >34.1532</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row0_col4" class="data row0 col4" >-8.77411</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row0_col5" class="data row0 col5" >-14.3976</td> 
    </tr>    <tr> 
        <th id="T_848b4a09_9564_11e8_a1ed_525400123456level0_row1" class="row_heading level0 row1" >BZIP2o</th> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row1_col0" class="data row1 col0" >57.4523</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row1_col1" class="data row1 col1" >0</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row1_col2" class="data row1 col2" >92.2603</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row1_col3" class="data row1 col3" >110.527</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row1_col4" class="data row1 col4" >43.6394</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row1_col5" class="data row1 col5" >34.779</td> 
    </tr>    <tr> 
        <th id="T_848b4a09_9564_11e8_a1ed_525400123456level0_row2" class="row_heading level0 row2" >LZMAo</th> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row2_col0" class="data row2 col0" >-18.1814</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row2_col1" class="data row2 col1" >-46.7408</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row2_col2" class="data row2 col2" >0</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row2_col3" class="data row2 col3" >9.75262</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row2_col4" class="data row2 col4" >-25.3562</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row2_col5" class="data row2 col5" >-29.9561</td> 
    </tr>    <tr> 
        <th id="T_848b4a09_9564_11e8_a1ed_525400123456level0_row3" class="row_heading level0 row3" >XZo</th> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row3_col0" class="data row3 col0" >-25.2083</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row3_col1" class="data row3 col1" >-51.4114</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row3_col2" class="data row3 col2" >-8.58532</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row3_col3" class="data row3 col3" >0</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row3_col4" class="data row3 col4" >-31.7711</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row3_col5" class="data row3 col5" >-35.9723</td> 
    </tr>    <tr> 
        <th id="T_848b4a09_9564_11e8_a1ed_525400123456level0_row4" class="row_heading level0 row4" >LZOo</th> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row4_col0" class="data row4 col0" >9.62686</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row4_col1" class="data row4 col1" >-28.5649</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row4_col2" class="data row4 col2" >34.0124</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row4_col3" class="data row4 col3" >47.066</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row4_col4" class="data row4 col4" >0</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row4_col5" class="data row4 col5" >-6.16775</td> 
    </tr>    <tr> 
        <th id="T_848b4a09_9564_11e8_a1ed_525400123456level0_row5" class="row_heading level0 row5" >LZ4o</th> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row5_col0" class="data row5 col0" >16.84</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row5_col1" class="data row5 col1" >-23.8675</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row5_col2" class="data row5 col2" >42.8332</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row5_col3" class="data row5 col3" >56.7546</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row5_col4" class="data row5 col4" >6.5758</td> 
        <td id="T_848b4a09_9564_11e8_a1ed_525400123456row5_col5" class="data row5 col5" >0</td> 
    </tr></tbody> 
</table> 




```python
compareCompress("-vmlinux").style.set_caption('Difference (average in percentage) per compression methods, vmlinux').applymap(color_negative_positive)

```




<style  type="text/css" >
    #T_848b4a0a_9564_11e8_a1ed_525400123456row0_col0 {
            color:  black;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row0_col1 {
            color:  red;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row0_col2 {
            color:  green;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row0_col3 {
            color:  green;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row0_col4 {
            color:  red;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row0_col5 {
            color:  red;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row1_col0 {
            color:  green;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row1_col1 {
            color:  black;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row1_col2 {
            color:  green;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row1_col3 {
            color:  green;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row1_col4 {
            color:  green;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row1_col5 {
            color:  green;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row2_col0 {
            color:  red;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row2_col1 {
            color:  red;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row2_col2 {
            color:  black;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row2_col3 {
            color:  green;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row2_col4 {
            color:  red;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row2_col5 {
            color:  red;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row3_col0 {
            color:  red;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row3_col1 {
            color:  red;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row3_col2 {
            color:  red;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row3_col3 {
            color:  black;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row3_col4 {
            color:  red;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row3_col5 {
            color:  red;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row4_col0 {
            color:  green;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row4_col1 {
            color:  red;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row4_col2 {
            color:  green;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row4_col3 {
            color:  green;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row4_col4 {
            color:  black;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row4_col5 {
            color:  red;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row5_col0 {
            color:  green;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row5_col1 {
            color:  red;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row5_col2 {
            color:  green;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row5_col3 {
            color:  green;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row5_col4 {
            color:  green;
        }    #T_848b4a0a_9564_11e8_a1ed_525400123456row5_col5 {
            color:  black;
        }</style>  
<table id="T_848b4a0a_9564_11e8_a1ed_525400123456" ><caption>Difference (average in percentage) per compression methods, vmlinux</caption> 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >GZIP</th> 
        <th class="col_heading level0 col1" >BZIP2</th> 
        <th class="col_heading level0 col2" >LZMA</th> 
        <th class="col_heading level0 col3" >XZ</th> 
        <th class="col_heading level0 col4" >LZO</th> 
        <th class="col_heading level0 col5" >LZ4</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_848b4a0a_9564_11e8_a1ed_525400123456level0_row0" class="row_heading level0 row0" >GZIPo</th> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row0_col0" class="data row0 col0" >0</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row0_col1" class="data row0 col1" >-28.9643</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row0_col2" class="data row0 col2" >16.4638</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row0_col3" class="data row0 col3" >24.7346</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row0_col4" class="data row0 col4" >-6.91201</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row0_col5" class="data row0 col5" >-11.4933</td> 
    </tr>    <tr> 
        <th id="T_848b4a0a_9564_11e8_a1ed_525400123456level0_row1" class="row_heading level0 row1" >BZIP2o</th> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row1_col0" class="data row1 col0" >42.3618</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row1_col1" class="data row1 col1" >0</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row1_col2" class="data row1 col2" >65.4498</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row1_col3" class="data row1 col3" >76.894</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row1_col4" class="data row1 col4" >32.6119</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row1_col5" class="data row1 col5" >26.1396</td> 
    </tr>    <tr> 
        <th id="T_848b4a0a_9564_11e8_a1ed_525400123456level0_row2" class="row_heading level0 row2" >LZMAo</th> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row2_col0" class="data row2 col0" >-14.0965</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row2_col1" class="data row2 col1" >-39.1051</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row2_col2" class="data row2 col2" >0</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row2_col3" class="data row2 col3" >7.07283</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row2_col4" class="data row2 col4" >-20.0205</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row2_col5" class="data row2 col5" >-23.9489</td> 
    </tr>    <tr> 
        <th id="T_848b4a0a_9564_11e8_a1ed_525400123456level0_row3" class="row_heading level0 row3" >XZo</th> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row3_col0" class="data row3 col0" >-19.5922</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row3_col1" class="data row3 col1" >-43.0945</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row3_col2" class="data row3 col2" >-6.42222</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row3_col3" class="data row3 col3" >0</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row3_col4" class="data row3 col4" >-25.1318</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row3_col5" class="data row3 col5" >-28.8019</td> 
    </tr>    <tr> 
        <th id="T_848b4a0a_9564_11e8_a1ed_525400123456level0_row4" class="row_heading level0 row4" >LZOo</th> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row4_col0" class="data row4 col0" >7.43603</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row4_col1" class="data row4 col1" >-23.6317</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row4_col2" class="data row4 col2" >25.1457</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row4_col3" class="data row4 col3" >34.0424</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row4_col4" class="data row4 col4" >0</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row4_col5" class="data row4 col5" >-4.92661</td> 
    </tr>    <tr> 
        <th id="T_848b4a0a_9564_11e8_a1ed_525400123456level0_row5" class="row_heading level0 row5" >LZ4o</th> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row5_col0" class="data row5 col0" >13.0136</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row5_col1" class="data row5 col1" >-19.6333</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row5_col2" class="data row5 col2" >31.6561</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row5_col3" class="data row5 col3" >41.0305</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row5_col4" class="data row5 col4" >5.18566</td> 
        <td id="T_848b4a0a_9564_11e8_a1ed_525400123456row5_col5" class="data row5 col5" >0</td> 
    </tr></tbody> 
</table> 




```python
cm = sns.light_palette("green", as_cmap=True)
pd.DataFrame.corr(rawtuxdata[size_methods]).style.set_caption('Correlations between size measures').background_gradient(cmap=cm)

```




<style  type="text/css" >
    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col0 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col1 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col2 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col3 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col4 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col5 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col6 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col7 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col8 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col9 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col10 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col11 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col12 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col13 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col14 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col15 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col16 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col17 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row0_col18 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col0 {
            background-color:  #d1f4d1;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col1 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col2 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col3 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col4 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col5 {
            background-color:  #058305;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col6 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col7 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col8 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col9 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col10 {
            background-color:  #138a13;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col11 {
            background-color:  #138a13;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col12 {
            background-color:  #128a12;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col13 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col14 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col15 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col16 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col17 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row1_col18 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col0 {
            background-color:  #d1f4d1;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col1 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col2 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col3 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col4 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col5 {
            background-color:  #058305;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col6 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col7 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col8 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col9 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col10 {
            background-color:  #138a13;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col11 {
            background-color:  #138a13;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col12 {
            background-color:  #128a12;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col13 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col14 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col15 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col16 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col17 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row2_col18 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col0 {
            background-color:  #d3f5d3;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col1 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col2 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col3 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col4 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col5 {
            background-color:  #058305;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col6 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col7 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col8 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col9 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col10 {
            background-color:  #138a13;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col11 {
            background-color:  #138a13;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col12 {
            background-color:  #128a12;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col13 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col14 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col15 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col16 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col17 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row3_col18 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col0 {
            background-color:  #c2ecc2;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col1 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col2 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col3 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col4 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col5 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col6 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col7 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col8 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col9 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col10 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col11 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col12 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col13 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col14 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col15 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col16 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col17 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row4_col18 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col0 {
            background-color:  #c4edc4;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col1 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col2 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col3 {
            background-color:  #058305;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col4 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col5 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col6 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col7 {
            background-color:  #058305;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col8 {
            background-color:  #058305;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col9 {
            background-color:  #068306;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col10 {
            background-color:  #198e19;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col11 {
            background-color:  #198e19;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col12 {
            background-color:  #198e19;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col13 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col14 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col15 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col16 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col17 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row5_col18 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col0 {
            background-color:  #c4edc4;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col1 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col2 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col3 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col4 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col5 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col6 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col7 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col8 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col9 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col10 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col11 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col12 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col13 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col14 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col15 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col16 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col17 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row6_col18 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col0 {
            background-color:  #dbf9db;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col1 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col2 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col3 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col4 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col5 {
            background-color:  #068306;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col6 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col7 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col8 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col9 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col10 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col11 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col12 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col13 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col14 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col15 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col16 {
            background-color:  #048204;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col17 {
            background-color:  #048204;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row7_col18 {
            background-color:  #038103;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col0 {
            background-color:  #dbf9db;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col1 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col2 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col3 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col4 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col5 {
            background-color:  #068306;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col6 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col7 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col8 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col9 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col10 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col11 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col12 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col13 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col14 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col15 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col16 {
            background-color:  #048204;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col17 {
            background-color:  #048204;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row8_col18 {
            background-color:  #038103;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col0 {
            background-color:  #defbde;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col1 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col2 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col3 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col4 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col5 {
            background-color:  #078407;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col6 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col7 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col8 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col9 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col10 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col11 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col12 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col13 {
            background-color:  #038103;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col14 {
            background-color:  #038103;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col15 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col16 {
            background-color:  #048204;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col17 {
            background-color:  #048204;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row9_col18 {
            background-color:  #048204;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col0 {
            background-color:  #e3fee3;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col1 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col2 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col3 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col4 {
            background-color:  #188d18;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col5 {
            background-color:  #1d901d;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col6 {
            background-color:  #188d18;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col7 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col8 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col9 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col10 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col11 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col12 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col13 {
            background-color:  #178c17;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col14 {
            background-color:  #178c17;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col15 {
            background-color:  #168c16;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col16 {
            background-color:  #198e19;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col17 {
            background-color:  #198e19;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row10_col18 {
            background-color:  #198e19;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col0 {
            background-color:  #e2fde2;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col1 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col2 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col3 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col4 {
            background-color:  #188d18;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col5 {
            background-color:  #1d901d;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col6 {
            background-color:  #188d18;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col7 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col8 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col9 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col10 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col11 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col12 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col13 {
            background-color:  #178c17;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col14 {
            background-color:  #178c17;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col15 {
            background-color:  #168c16;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col16 {
            background-color:  #198e19;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col17 {
            background-color:  #198e19;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row11_col18 {
            background-color:  #198e19;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col0 {
            background-color:  #e5ffe5;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col1 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col2 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col3 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col4 {
            background-color:  #198e19;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col5 {
            background-color:  #1d901d;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col6 {
            background-color:  #188d18;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col7 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col8 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col9 {
            background-color:  #148b14;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col10 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col11 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col12 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col13 {
            background-color:  #178c17;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col14 {
            background-color:  #178c17;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col15 {
            background-color:  #178c17;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col16 {
            background-color:  #1a8e1a;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col17 {
            background-color:  #1a8e1a;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row12_col18 {
            background-color:  #198e19;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col0 {
            background-color:  #c0eac0;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col1 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col2 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col3 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col4 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col5 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col6 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col7 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col8 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col9 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col10 {
            background-color:  #138a13;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col11 {
            background-color:  #138a13;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col12 {
            background-color:  #138a13;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col13 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col14 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col15 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col16 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col17 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row13_col18 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col0 {
            background-color:  #bfeabf;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col1 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col2 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col3 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col4 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col5 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col6 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col7 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col8 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col9 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col10 {
            background-color:  #138a13;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col11 {
            background-color:  #138a13;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col12 {
            background-color:  #138a13;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col13 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col14 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col15 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col16 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col17 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row14_col18 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col0 {
            background-color:  #c2ebc2;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col1 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col2 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col3 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col4 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col5 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col6 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col7 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col8 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col9 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col10 {
            background-color:  #138a13;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col11 {
            background-color:  #138a13;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col12 {
            background-color:  #138a13;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col13 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col14 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col15 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col16 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col17 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row15_col18 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col0 {
            background-color:  #b9e7b9;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col1 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col2 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col3 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col4 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col5 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col6 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col7 {
            background-color:  #038103;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col8 {
            background-color:  #038103;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col9 {
            background-color:  #048204;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col10 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col11 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col12 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col13 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col14 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col15 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col16 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col17 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row16_col18 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col0 {
            background-color:  #b8e6b8;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col1 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col2 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col3 {
            background-color:  #028102;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col4 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col5 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col6 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col7 {
            background-color:  #038103;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col8 {
            background-color:  #038103;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col9 {
            background-color:  #048204;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col10 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col11 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col12 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col13 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col14 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col15 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col16 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col17 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row17_col18 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col0 {
            background-color:  #bbe8bb;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col1 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col2 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col3 {
            background-color:  #018001;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col4 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col5 {
            background-color:  #058205;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col6 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col7 {
            background-color:  #038103;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col8 {
            background-color:  #038103;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col9 {
            background-color:  #038103;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col10 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col11 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col12 {
            background-color:  #158b15;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col13 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col14 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col15 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col16 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col17 {
            background-color:  #008000;
        }    #T_848b4a0b_9564_11e8_a1ed_525400123456row18_col18 {
            background-color:  #008000;
        }</style>  
<table id="T_848b4a0b_9564_11e8_a1ed_525400123456" ><caption>Correlations between size measures</caption> 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >vmlinux</th> 
        <th class="col_heading level0 col1" >GZIP-bzImage</th> 
        <th class="col_heading level0 col2" >GZIP-vmlinux</th> 
        <th class="col_heading level0 col3" >GZIP</th> 
        <th class="col_heading level0 col4" >BZIP2-bzImage</th> 
        <th class="col_heading level0 col5" >BZIP2-vmlinux</th> 
        <th class="col_heading level0 col6" >BZIP2</th> 
        <th class="col_heading level0 col7" >LZMA-bzImage</th> 
        <th class="col_heading level0 col8" >LZMA-vmlinux</th> 
        <th class="col_heading level0 col9" >LZMA</th> 
        <th class="col_heading level0 col10" >XZ-bzImage</th> 
        <th class="col_heading level0 col11" >XZ-vmlinux</th> 
        <th class="col_heading level0 col12" >XZ</th> 
        <th class="col_heading level0 col13" >LZO-bzImage</th> 
        <th class="col_heading level0 col14" >LZO-vmlinux</th> 
        <th class="col_heading level0 col15" >LZO</th> 
        <th class="col_heading level0 col16" >LZ4-bzImage</th> 
        <th class="col_heading level0 col17" >LZ4-vmlinux</th> 
        <th class="col_heading level0 col18" >LZ4</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row0" class="row_heading level0 row0" >vmlinux</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col0" class="data row0 col0" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col1" class="data row0 col1" >0.896046</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col2" class="data row0 col2" >0.896099</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col3" class="data row0 col3" >0.894793</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col4" class="data row0 col4" >0.903326</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col5" class="data row0 col5" >0.902387</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col6" class="data row0 col6" >0.902123</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col7" class="data row0 col7" >0.89084</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col8" class="data row0 col8" >0.890907</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col9" class="data row0 col9" >0.889243</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col10" class="data row0 col10" >0.887221</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col11" class="data row0 col11" >0.887295</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col12" class="data row0 col12" >0.885469</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col13" class="data row0 col13" >0.90467</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col14" class="data row0 col14" >0.904715</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col15" class="data row0 col15" >0.903608</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col16" class="data row0 col16" >0.9078</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col17" class="data row0 col17" >0.907841</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row0_col18" class="data row0 col18" >0.906841</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row1" class="row_heading level0 row1" >GZIP-bzImage</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col0" class="data row1 col0" >0.896046</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col1" class="data row1 col1" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col2" class="data row1 col2" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col3" class="data row1 col3" >0.999989</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col4" class="data row1 col4" >0.999533</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col5" class="data row1 col5" >0.997573</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col6" class="data row1 col6" >0.99963</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col7" class="data row1 col7" >0.999725</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col8" class="data row1 col8" >0.999726</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col9" class="data row1 col9" >0.999678</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col10" class="data row1 col10" >0.990732</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col11" class="data row1 col11" >0.990732</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col12" class="data row1 col12" >0.990704</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col13" class="data row1 col13" >0.999669</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col14" class="data row1 col14" >0.999666</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col15" class="data row1 col15" >0.999739</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col16" class="data row1 col16" >0.999271</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col17" class="data row1 col17" >0.999266</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row1_col18" class="data row1 col18" >0.999373</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row2" class="row_heading level0 row2" >GZIP-vmlinux</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col0" class="data row2 col0" >0.896099</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col1" class="data row2 col1" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col2" class="data row2 col2" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col3" class="data row2 col3" >0.999988</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col4" class="data row2 col4" >0.999538</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col5" class="data row2 col5" >0.997578</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col6" class="data row2 col6" >0.999633</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col7" class="data row2 col7" >0.999725</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col8" class="data row2 col8" >0.999726</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col9" class="data row2 col9" >0.999676</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col10" class="data row2 col10" >0.990732</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col11" class="data row2 col11" >0.990732</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col12" class="data row2 col12" >0.990703</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col13" class="data row2 col13" >0.999672</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col14" class="data row2 col14" >0.999669</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col15" class="data row2 col15" >0.999741</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col16" class="data row2 col16" >0.999277</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col17" class="data row2 col17" >0.999272</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row2_col18" class="data row2 col18" >0.999377</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row3" class="row_heading level0 row3" >GZIP</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col0" class="data row3 col0" >0.894793</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col1" class="data row3 col1" >0.999989</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col2" class="data row3 col2" >0.999988</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col3" class="data row3 col3" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col4" class="data row3 col4" >0.999418</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col5" class="data row3 col5" >0.997452</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col6" class="data row3 col6" >0.999537</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col7" class="data row3 col7" >0.999738</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col8" class="data row3 col8" >0.999737</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col9" class="data row3 col9" >0.999718</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col10" class="data row3 col10" >0.990712</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col11" class="data row3 col11" >0.99071</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col12" class="data row3 col12" >0.990715</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col13" class="data row3 col13" >0.999571</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col14" class="data row3 col14" >0.999567</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col15" class="data row3 col15" >0.999661</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col16" class="data row3 col16" >0.99913</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col17" class="data row3 col17" >0.999125</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row3_col18" class="data row3 col18" >0.999251</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row4" class="row_heading level0 row4" >BZIP2-bzImage</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col0" class="data row4 col0" >0.903326</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col1" class="data row4 col1" >0.999533</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col2" class="data row4 col2" >0.999538</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col3" class="data row4 col3" >0.999418</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col4" class="data row4 col4" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col5" class="data row4 col5" >0.998085</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col6" class="data row4 col6" >0.999988</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col7" class="data row4 col7" >0.999057</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col8" class="data row4 col8" >0.999063</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col9" class="data row4 col9" >0.998879</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col10" class="data row4 col10" >0.989574</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col11" class="data row4 col11" >0.989579</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col12" class="data row4 col12" >0.989399</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col13" class="data row4 col13" >0.999818</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col14" class="data row4 col14" >0.999818</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col15" class="data row4 col15" >0.999793</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col16" class="data row4 col16" >0.999772</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col17" class="data row4 col17" >0.999771</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row4_col18" class="data row4 col18" >0.999785</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row5" class="row_heading level0 row5" >BZIP2-vmlinux</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col0" class="data row5 col0" >0.902387</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col1" class="data row5 col1" >0.997573</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col2" class="data row5 col2" >0.997578</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col3" class="data row5 col3" >0.997452</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col4" class="data row5 col4" >0.998085</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col5" class="data row5 col5" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col6" class="data row5 col6" >0.998067</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col7" class="data row5 col7" >0.99709</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col8" class="data row5 col8" >0.997098</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col9" class="data row5 col9" >0.996906</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col10" class="data row5 col10" >0.987649</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col11" class="data row5 col11" >0.987656</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col12" class="data row5 col12" >0.987467</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col13" class="data row5 col13" >0.997896</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col14" class="data row5 col14" >0.997897</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col15" class="data row5 col15" >0.997866</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col16" class="data row5 col16" >0.997862</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col17" class="data row5 col17" >0.997862</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row5_col18" class="data row5 col18" >0.997871</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row6" class="row_heading level0 row6" >BZIP2</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col0" class="data row6 col0" >0.902123</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col1" class="data row6 col1" >0.99963</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col2" class="data row6 col2" >0.999633</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col3" class="data row6 col3" >0.999537</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col4" class="data row6 col4" >0.999988</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col5" class="data row6 col5" >0.998067</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col6" class="data row6 col6" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col7" class="data row6 col7" >0.999177</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col8" class="data row6 col8" >0.999182</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col9" class="data row6 col9" >0.999028</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col10" class="data row6 col10" >0.98966</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col11" class="data row6 col11" >0.989664</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col12" class="data row6 col12" >0.989518</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col13" class="data row6 col13" >0.999824</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col14" class="data row6 col14" >0.999824</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col15" class="data row6 col15" >0.99982</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col16" class="data row6 col16" >0.999734</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col17" class="data row6 col17" >0.999732</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row6_col18" class="data row6 col18" >0.999767</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row7" class="row_heading level0 row7" >LZMA-bzImage</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col0" class="data row7 col0" >0.89084</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col1" class="data row7 col1" >0.999725</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col2" class="data row7 col2" >0.999725</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col3" class="data row7 col3" >0.999738</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col4" class="data row7 col4" >0.999057</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col5" class="data row7 col5" >0.99709</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col6" class="data row7 col6" >0.999177</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col7" class="data row7 col7" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col8" class="data row7 col8" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col9" class="data row7 col9" >0.999983</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col10" class="data row7 col10" >0.990084</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col11" class="data row7 col11" >0.990083</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col12" class="data row7 col12" >0.990088</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col13" class="data row7 col13" >0.999006</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col14" class="data row7 col14" >0.999002</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col15" class="data row7 col15" >0.999097</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col16" class="data row7 col16" >0.99845</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col17" class="data row7 col17" >0.998445</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row7_col18" class="data row7 col18" >0.998571</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row8" class="row_heading level0 row8" >LZMA-vmlinux</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col0" class="data row8 col0" >0.890907</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col1" class="data row8 col1" >0.999726</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col2" class="data row8 col2" >0.999726</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col3" class="data row8 col3" >0.999737</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col4" class="data row8 col4" >0.999063</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col5" class="data row8 col5" >0.997098</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col6" class="data row8 col6" >0.999182</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col7" class="data row8 col7" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col8" class="data row8 col8" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col9" class="data row8 col9" >0.999981</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col10" class="data row8 col10" >0.990086</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col11" class="data row8 col11" >0.990084</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col12" class="data row8 col12" >0.990088</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col13" class="data row8 col13" >0.999011</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col14" class="data row8 col14" >0.999008</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col15" class="data row8 col15" >0.999101</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col16" class="data row8 col16" >0.998458</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col17" class="data row8 col17" >0.998453</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row8_col18" class="data row8 col18" >0.998578</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row9" class="row_heading level0 row9" >LZMA</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col0" class="data row9 col0" >0.889243</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col1" class="data row9 col1" >0.999678</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col2" class="data row9 col2" >0.999676</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col3" class="data row9 col3" >0.999718</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col4" class="data row9 col4" >0.998879</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col5" class="data row9 col5" >0.996906</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col6" class="data row9 col6" >0.999028</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col7" class="data row9 col7" >0.999983</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col8" class="data row9 col8" >0.999981</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col9" class="data row9 col9" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col10" class="data row9 col10" >0.990024</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col11" class="data row9 col11" >0.990021</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col12" class="data row9 col12" >0.990067</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col13" class="data row9 col13" >0.99885</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col14" class="data row9 col14" >0.998845</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col15" class="data row9 col15" >0.998966</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col16" class="data row9 col16" >0.998241</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col17" class="data row9 col17" >0.998234</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row9_col18" class="data row9 col18" >0.998385</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row10" class="row_heading level0 row10" >XZ-bzImage</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col0" class="data row10 col0" >0.887221</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col1" class="data row10 col1" >0.990732</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col2" class="data row10 col2" >0.990732</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col3" class="data row10 col3" >0.990712</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col4" class="data row10 col4" >0.989574</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col5" class="data row10 col5" >0.987649</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col6" class="data row10 col6" >0.98966</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col7" class="data row10 col7" >0.990084</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col8" class="data row10 col8" >0.990086</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col9" class="data row10 col9" >0.990024</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col10" class="data row10 col10" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col11" class="data row10 col11" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col12" class="data row10 col12" >0.999978</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col13" class="data row10 col13" >0.990578</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col14" class="data row10 col14" >0.990576</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col15" class="data row10 col15" >0.990639</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col16" class="data row10 col16" >0.989711</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col17" class="data row10 col17" >0.989707</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row10_col18" class="data row10 col18" >0.989804</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row11" class="row_heading level0 row11" >XZ-vmlinux</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col0" class="data row11 col0" >0.887295</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col1" class="data row11 col1" >0.990732</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col2" class="data row11 col2" >0.990732</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col3" class="data row11 col3" >0.99071</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col4" class="data row11 col4" >0.989579</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col5" class="data row11 col5" >0.987656</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col6" class="data row11 col6" >0.989664</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col7" class="data row11 col7" >0.990083</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col8" class="data row11 col8" >0.990084</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col9" class="data row11 col9" >0.990021</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col10" class="data row11 col10" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col11" class="data row11 col11" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col12" class="data row11 col12" >0.999976</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col13" class="data row11 col13" >0.990583</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col14" class="data row11 col14" >0.99058</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col15" class="data row11 col15" >0.990642</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col16" class="data row11 col16" >0.989719</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col17" class="data row11 col17" >0.989715</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row11_col18" class="data row11 col18" >0.98981</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row12" class="row_heading level0 row12" >XZ</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col0" class="data row12 col0" >0.885469</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col1" class="data row12 col1" >0.990704</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col2" class="data row12 col2" >0.990703</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col3" class="data row12 col3" >0.990715</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col4" class="data row12 col4" >0.989399</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col5" class="data row12 col5" >0.987467</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col6" class="data row12 col6" >0.989518</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col7" class="data row12 col7" >0.990088</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col8" class="data row12 col8" >0.990088</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col9" class="data row12 col9" >0.990067</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col10" class="data row12 col10" >0.999978</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col11" class="data row12 col11" >0.999976</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col12" class="data row12 col12" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col13" class="data row12 col13" >0.99043</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col14" class="data row12 col14" >0.990425</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col15" class="data row12 col15" >0.990518</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col16" class="data row12 col16" >0.989502</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col17" class="data row12 col17" >0.989497</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row12_col18" class="data row12 col18" >0.989622</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row13" class="row_heading level0 row13" >LZO-bzImage</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col0" class="data row13 col0" >0.90467</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col1" class="data row13 col1" >0.999669</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col2" class="data row13 col2" >0.999672</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col3" class="data row13 col3" >0.999571</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col4" class="data row13 col4" >0.999818</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col5" class="data row13 col5" >0.997896</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col6" class="data row13 col6" >0.999824</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col7" class="data row13 col7" >0.999006</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col8" class="data row13 col8" >0.999011</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col9" class="data row13 col9" >0.99885</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col10" class="data row13 col10" >0.990578</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col11" class="data row13 col11" >0.990583</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col12" class="data row13 col12" >0.99043</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col13" class="data row13 col13" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col14" class="data row13 col14" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col15" class="data row13 col15" >0.999991</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col16" class="data row13 col16" >0.999907</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col17" class="data row13 col17" >0.999906</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row13_col18" class="data row13 col18" >0.999936</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row14" class="row_heading level0 row14" >LZO-vmlinux</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col0" class="data row14 col0" >0.904715</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col1" class="data row14 col1" >0.999666</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col2" class="data row14 col2" >0.999669</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col3" class="data row14 col3" >0.999567</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col4" class="data row14 col4" >0.999818</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col5" class="data row14 col5" >0.997897</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col6" class="data row14 col6" >0.999824</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col7" class="data row14 col7" >0.999002</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col8" class="data row14 col8" >0.999008</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col9" class="data row14 col9" >0.998845</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col10" class="data row14 col10" >0.990576</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col11" class="data row14 col11" >0.99058</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col12" class="data row14 col12" >0.990425</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col13" class="data row14 col13" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col14" class="data row14 col14" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col15" class="data row14 col15" >0.99999</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col16" class="data row14 col16" >0.999909</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col17" class="data row14 col17" >0.999907</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row14_col18" class="data row14 col18" >0.999936</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row15" class="row_heading level0 row15" >LZO</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col0" class="data row15 col0" >0.903608</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col1" class="data row15 col1" >0.999739</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col2" class="data row15 col2" >0.999741</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col3" class="data row15 col3" >0.999661</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col4" class="data row15 col4" >0.999793</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col5" class="data row15 col5" >0.997866</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col6" class="data row15 col6" >0.99982</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col7" class="data row15 col7" >0.999097</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col8" class="data row15 col8" >0.999101</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col9" class="data row15 col9" >0.998966</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col10" class="data row15 col10" >0.990639</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col11" class="data row15 col11" >0.990642</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col12" class="data row15 col12" >0.990518</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col13" class="data row15 col13" >0.999991</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col14" class="data row15 col14" >0.99999</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col15" class="data row15 col15" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col16" class="data row15 col16" >0.999859</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col17" class="data row15 col17" >0.999857</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row15_col18" class="data row15 col18" >0.999905</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row16" class="row_heading level0 row16" >LZ4-bzImage</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col0" class="data row16 col0" >0.9078</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col1" class="data row16 col1" >0.999271</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col2" class="data row16 col2" >0.999277</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col3" class="data row16 col3" >0.99913</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col4" class="data row16 col4" >0.999772</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col5" class="data row16 col5" >0.997862</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col6" class="data row16 col6" >0.999734</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col7" class="data row16 col7" >0.99845</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col8" class="data row16 col8" >0.998458</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col9" class="data row16 col9" >0.998241</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col10" class="data row16 col10" >0.989711</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col11" class="data row16 col11" >0.989719</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col12" class="data row16 col12" >0.989502</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col13" class="data row16 col13" >0.999907</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col14" class="data row16 col14" >0.999909</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col15" class="data row16 col15" >0.999859</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col16" class="data row16 col16" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col17" class="data row16 col17" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row16_col18" class="data row16 col18" >0.999992</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row17" class="row_heading level0 row17" >LZ4-vmlinux</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col0" class="data row17 col0" >0.907841</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col1" class="data row17 col1" >0.999266</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col2" class="data row17 col2" >0.999272</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col3" class="data row17 col3" >0.999125</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col4" class="data row17 col4" >0.999771</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col5" class="data row17 col5" >0.997862</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col6" class="data row17 col6" >0.999732</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col7" class="data row17 col7" >0.998445</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col8" class="data row17 col8" >0.998453</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col9" class="data row17 col9" >0.998234</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col10" class="data row17 col10" >0.989707</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col11" class="data row17 col11" >0.989715</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col12" class="data row17 col12" >0.989497</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col13" class="data row17 col13" >0.999906</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col14" class="data row17 col14" >0.999907</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col15" class="data row17 col15" >0.999857</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col16" class="data row17 col16" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col17" class="data row17 col17" >1</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row17_col18" class="data row17 col18" >0.999991</td> 
    </tr>    <tr> 
        <th id="T_848b4a0b_9564_11e8_a1ed_525400123456level0_row18" class="row_heading level0 row18" >LZ4</th> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col0" class="data row18 col0" >0.906841</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col1" class="data row18 col1" >0.999373</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col2" class="data row18 col2" >0.999377</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col3" class="data row18 col3" >0.999251</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col4" class="data row18 col4" >0.999785</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col5" class="data row18 col5" >0.997871</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col6" class="data row18 col6" >0.999767</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col7" class="data row18 col7" >0.998571</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col8" class="data row18 col8" >0.998578</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col9" class="data row18 col9" >0.998385</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col10" class="data row18 col10" >0.989804</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col11" class="data row18 col11" >0.98981</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col12" class="data row18 col12" >0.989622</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col13" class="data row18 col13" >0.999936</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col14" class="data row18 col14" >0.999936</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col15" class="data row18 col15" >0.999905</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col16" class="data row18 col16" >0.999992</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col17" class="data row18 col17" >0.999991</td> 
        <td id="T_848b4a0b_9564_11e8_a1ed_525400123456row18_col18" class="data row18 col18" >1</td> 
    </tr></tbody> 
</table> 




```python
#from category_encoders import *
from sklearn.preprocessing import *

## class to integer encoding (y, n, m)

## note: we also remove non-tristate-options
# "in place" is to avoid memory burden (having two dfs in memory)

# encode labels with value between 0 and n_classes-1.
le = LabelEncoder()
# 2/3. FIT AND TRANSFORM
vml = rawtuxdata[size_methods]

# we remove non tri state options, but TODO there are perhaps some interesting options (numerical or string) here
#tuxdata = rawtuxdata.drop(columns=non_tristate_options).drop(columns=['vmlinux']).apply(le.fit_transform)
rawtuxdata = rawtuxdata.drop(columns=non_tristate_options).drop(columns=size_methods).apply(le.fit_transform)

#tuxdata['vmlinux'] = vml 
rawtuxdata[size_methods] = vml
 
rawtuxdata.shape, rawtuxdata.info(memory_usage='deep')
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2489 entries, 0 to 2499
    Columns: 9230 entries, OPENVSWITCH to LZ4
    dtypes: int64(9230)
    memory usage: 175.3 MB





    ((2489, 9230), None)




```python
#### takes a while
# One-Hot-Encoding 
#from sklearn.preprocessing import *

#enc = OneHotEncoder()
#o_sizes = rawtuxdata[size_methods]
#oh_tuxdata = enc.fit_transform(rawtuxdata)
#oh_tuxdata.shape, o_sizes.shape
# rawtuxdata.drop(columns=non_tristate_options).drop(columns=size_methods).apply(enc.fit_transform)
#oh_tuxdata[size_methods] = o_sizes
```


```python
# DUMMY (with Pandas)

#o_sizes = tuxdata[size_methods]
#tuxdata_dummy = pd.get_dummies(rawtuxdata.drop(columns=size_methods), columns=rawtuxdata.drop(columns=size_methods).columns)
#tuxdata_dummy[size_methods] = o_sizes
#tuxdata_dummy.shape
```


```python
# Data exploration (again)
#print(rawtuxdata['UBSAN_SANITIZE_ALL'].value_counts(), rawtuxdata['COMPILE_TEST'].value_counts(), rawtuxdata['NOHIGHMEM'].value_counts(), rawtuxdata['OPTIMIZE_INLINING'].value_counts(), rawtuxdata['SLOB'].value_counts(), rawtuxdata['CC_OPTIMIZE_FOR_SIZE'].value_counts(), sep='\n')
```


```python
from enum import Enum
class LearningStrategy(Enum):
    LINEAR = 1
    AUTOML = 2
    ML = 3
```


```python
from sklearn.pipeline import Pipeline

# https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/
class PipelineRFE(Pipeline):

    def fit(self, X, y=None, **fit_params):
        super(PipelineRFE, self).fit(X, y, **fit_params)
        self.feature_importances_ = self.steps[-1][-1].feature_importances_
        return self
```


```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn import svm
from sklearn.decomposition import PCA
from tpot import TPOTRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.neural_network import MLPRegressor





# drop(columns=["date", "time", "vmlinux", "cid"])
# tuxdata.drop(columns=non_tristate_options)

NO_ENCODED_VALUE = le.transform(['n'])[0] 

def mkNoOption(option_name):
    return "(" + option_name + " == " + str(NO_ENCODED_VALUE) + ")"

def prefilter_data(tuxdata):    
    return rawtuxdata
    #return rawtuxdata.query(mkNoOption("DEBUG_INFO"))
    #return rawtuxdata.query(mkNoOption("DEBUG_INFO") + " & " + mkNoOption("GCOV_PROFILE_ALL") + " & " + mkNoOption("KASAN") + " & " + mkNoOption("UBSAN_SANITIZE_ALL") + " & " + mkNoOption("RELOCATABLE") + " & " + mkNoOption("XFS_DEBUG"))
                

def regLearning(tuxdata, kindOfLearning=LearningStrategy.ML):
 
    TESTING_SIZE=0.2 # 0.9 means 10% for training, 90% for testing
    size_of_interest = "vmlinux" # could be LZO, BZIP, etc. 
    PRINT_FEATURE_IMPORTANCES = True
   
       
    #X_train, X_test, y_train, y_test = train_test_split(tuxdata[(tuxdata['DEBUG_INFO'] == le.transform(['n'])[0])].drop(columns=size_methods), tuxdata[(tuxdata['DEBUG_INFO'] == le.transform(['n'])[0])][size_of_interest], test_size=TESTING_SIZE, random_state=0)  
    print ("Warning: prefiltering on DEBUG_INFO=n GCOV_PROFILE_ALL=n KASAN=n ....")   
    X_train, X_test, y_train, y_test = train_test_split(prefilter_data(tuxdata).drop(columns=size_methods), prefilter_data(tuxdata)[size_of_interest], test_size=TESTING_SIZE, random_state=0)  
  
    # multi output
    #X_train, X_test, y_train, y_test = train_test_split(tuxdata.drop(columns=size_methods), tuxdata[size_methods], test_size=TESTING_SIZE, random_state=0)  

    # train_test_split(tuxdata.drop(columns=['vmlinux']), tuxdata['vmlinux'], test_size=TESTING_SIZE, random_state=0)  

    #clf = RandomForestRegressor(n_estimators=100) 

    if kindOfLearning == LearningStrategy.LINEAR:
        regr =  linear_model.Lasso() # svm.SVC(kernel='linear') # linear_model.Ridge(alpha=.1) #  # linear_model.Lasso() # linear_model.SGDRegressor() #LinearRegression() # SGDRegressor or linear_model.Lasso()
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)

    elif kindOfLearning == LearningStrategy.AUTOML:


        tpot_config = {

            'sklearn.linear_model.ElasticNetCV': {
                'l1_ratio': np.arange(0.0, 1.01, 0.05),
                'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            },

            'sklearn.ensemble.ExtraTreesRegressor': {
                'n_estimators': [100],
                'max_features': np.arange(0.05, 1.01, 0.05),
                'min_samples_split': range(2, 21),
                'min_samples_leaf': range(1, 21),
                'bootstrap': [True, False]
            },

            'sklearn.ensemble.GradientBoostingRegressor': {
                'n_estimators': [100],
                'loss': ["ls", "lad", "huber", "quantile"],
                'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
                'max_depth': range(1, 11),
                'min_samples_split': range(2, 21),
                'min_samples_leaf': range(1, 21),
                'subsample': np.arange(0.05, 1.01, 0.05),
                'max_features': np.arange(0.05, 1.01, 0.05),
                'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
            },

            'sklearn.ensemble.AdaBoostRegressor': {
                'n_estimators': [100],
                'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
                'loss': ["linear", "square", "exponential"],
                'max_depth': range(1, 11)
            },

            'sklearn.tree.DecisionTreeRegressor': {
                'max_depth': range(1, 11),
                'min_samples_split': range(2, 21),
                'min_samples_leaf': range(1, 21)
            },

            'sklearn.neighbors.KNeighborsRegressor': {
                'n_neighbors': range(1, 101),
                'weights': ["uniform", "distance"],
                'p': [1, 2]
            },

            'sklearn.linear_model.LassoLarsCV': {
                'normalize': [True, False]
            },

            'sklearn.svm.LinearSVR': {
                'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
                'dual': [True, False],
                'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
                'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
            },

            'sklearn.ensemble.RandomForestRegressor': {
                'n_estimators': [100],
                'max_features': np.arange(0.05, 1.01, 0.05),
                'min_samples_split': range(2, 21),
                'min_samples_leaf': range(1, 21),
                'bootstrap': [True, False]
            },

            'sklearn.linear_model.RidgeCV': {
            },

            'xgboost.XGBRegressor': {
                'n_estimators': [100],
                'max_depth': range(1, 11),
                'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
                'subsample': np.arange(0.05, 1.01, 0.05),
                'min_child_weight': range(1, 21),
                'nthread': [1]
            }     
        }

        tpot = TPOTRegressor(generations=5, population_size=50, verbosity=3, config_dict=tpot_config, scoring='neg_mean_absolute_error')
        tpot.fit(X_train, y_train)
        y_pred = tpot.predict(X_test)
        print(tpot.score(X_test, y_test))
        print(tpot.evaluated_individuals_)
        tpot.export('tpot_boston_pipeline.py')

    else:
        assert (kindOfLearning == LearningStrategy.ML)
        clf = GradientBoostingRegressor(n_estimators=100) #RandomForestRegressor(n_estimators=100) #GradientBoostingRegressor(n_estimators=100) # KNeighborsRegressor() #RandomForestRegressor(n_estimators=100) # linear_model.SGDRegressor(alpha=0.15, max_iter=200)
        # #LassoLarsCV() # MLPRegressor() # GradientBoostingRegressor(n_estimators=100) # ExtraTreesRegressor(n_estimators=100) #RandomForestRegressor(n_estimators=100) # ExtraTreesRegressor(n_estimators=100) #  #   GradientBoostingRegressor(n_estimators=100) # 
        # 
        #estimator = RandomForestRegressor(n_estimators=100) # RidgeCV(alphas=[1000.0]) # LassoCV(tol = 0.001) #   #  # RandomForestRegressor(n_estimators=100) #LassoCV() #RidgeCV(alphas=[2000.0]) # LassoCV()
        #clf = PipelineRFE([ # Pipeline([
        #  ('feature_selection', SelectFromModel(estimator)), # tol = 0.001
        #  ('regression', GradientBoostingRegressor(n_estimators=100))
        #])
        #clf = PipelineRFE([
          #('reduce_dim', PCA()),
        #  ('feature_selection', SelectFromModel(estimator)), # tol = 0.001
        #  ('regression', GradientBoostingRegressor(n_estimators=100))
        #])
        #clf = make_pipeline(
        #    StackingEstimator(estimator=LassoLarsCV(normalize=False)),
        #    StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.6500000000000001, min_samples_leaf=10, min_samples_split=2, n_estimators=100)),
        #    KNeighborsRegressor(n_neighbors=82, p=2, weights="distance")
        #)


        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if PRINT_FEATURE_IMPORTANCES:
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]    

            TOP_FT_IMPORTANCE=100
            print("Feature ranking: " + "top (" + str(TOP_FT_IMPORTANCE) + ")")
            for f in range(TOP_FT_IMPORTANCE): # len(indices)
                print("%d. feature %s %d (%f)" % (f + 1, tuxdata.columns[indices[f]], indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    #plt.figure()
    #plt.title("Feature importances for size of vmlinux")
    #plt.bar(range(tuxdata.shape[1]), importances[indices], color="r", align="center")
    #plt.xticks(range(tuxdata.shape[1]), indices)
    #plt.xlim([-1, tuxdata.shape[1]])
    #plt.show()
    mae = mean_absolute_error (y_test, y_pred)# , multioutput='raw_values')
    mse = mean_squared_error (y_test, y_pred) #, multioutput='raw_values') 
    r2 = r2_score (y_test, y_pred) #, multioutput='raw_values') 
    mre = mean_relative_error (y_test, y_pred)

    ONE_MEGABYTE = 1048576

    print("Prediction score (MAE): %.2f" % (mae / ONE_MEGABYTE))
    print("Prediction score (MSE): %.2f" % (mse / ONE_MEGABYTE))
    print("Prediction score (R2): %.2f" % (r2))
    print("Prediction score (MRE): %.2f" % (mre))
    return y_pred, y_test
    
pr, re = regLearning(rawtuxdata, LearningStrategy.ML)
#regLearning(tuxdata_dummy)

```

    Warning: prefiltering on DEBUG_INFO=n GCOV_PROFILE_ALL=n KASAN=n ....
    Feature ranking: top (100)
    1. feature KCOV_INSTRUMENT_ALL 6361 (0.040404)
    2. feature X86_VSMP 4538 (0.036252)
    3. feature X86_NEED_RELOCS 4304 (0.033578)
    4. feature LOCK_STAT 7383 (0.031304)
    5. feature MAXSMP 8256 (0.028682)
    6. feature RANDOMIZE_BASE 4299 (0.024389)
    7. feature KALLSYMS_ABSOLUTE_PERCPU 6097 (0.018515)
    8. feature STRICT_MODULE_RWX 9150 (0.018484)
    9. feature MODULES 5200 (0.016810)
    10. feature ACPI_VIDEO 6456 (0.014337)
    11. feature DYNAMIC_DEBUG 5251 (0.013886)
    12. feature XFS_FS 4059 (0.012021)
    13. feature USB 3126 (0.011926)
    14. feature GENERIC_TRACER 8508 (0.011778)
    15. feature BLK_MQ_PCI 2364 (0.011768)
    16. feature DRM_RADEON 2760 (0.010115)
    17. feature DST_CACHE 1259 (0.009270)
    18. feature CFG80211 8804 (0.008820)
    19. feature FB_SVGALIB 198 (0.008448)
    20. feature INTEL_GTT 1148 (0.008133)
    21. feature SND_SOC_MAX98090 8729 (0.008018)
    22. feature DRM_NOUVEAU 2405 (0.007998)
    23. feature DRM_AMDGPU 2783 (0.007928)
    24. feature EXPERT 1340 (0.007599)
    25. feature PRINTK 3353 (0.007507)
    26. feature VIDEOBUF_GEN 4108 (0.007485)
    27. feature SG_POOL 3977 (0.007314)
    28. feature RAID6_PQ 973 (0.007055)
    29. feature PRINTK_NMI 6120 (0.007020)
    30. feature INET_TUNNEL 1246 (0.006626)
    31. feature MMU_NOTIFIER 1775 (0.006247)
    32. feature SERIAL_MCTRL_GPIO 3994 (0.005905)
    33. feature USB_CONFIGFS_MASS_STORAGE 8360 (0.005817)
    34. feature XPS 549 (0.005784)
    35. feature MAY_USE_DEVLINK 5100 (0.005762)
    36. feature MAC80211 7543 (0.005663)
    37. feature SCSI_ISCSI_ATTRS 222 (0.005441)
    38. feature WEXT_PROC 7038 (0.005298)
    39. feature MII 3809 (0.005173)
    40. feature BLK_SCSI_REQUEST 804 (0.004969)
    41. feature EVENT_TRACING 8420 (0.004922)
    42. feature OCFS2_FS 4388 (0.004703)
    43. feature SND_SOC_WM5110 3475 (0.004544)
    44. feature GENERIC_NET_UTILS 146 (0.004484)
    45. feature CEPH_FS 1359 (0.004482)
    46. feature SCHEDSTATS 6778 (0.004457)
    47. feature CONTEXT_SWITCH_TRACER 8501 (0.004452)
    48. feature SND_SOC_SSM4567 430 (0.004404)
    49. feature PM_GENERIC_DOMAINS_OF 5605 (0.004343)
    50. feature IMX_THERMAL 6512 (0.004250)
    51. feature TRACEPOINTS 7542 (0.004191)
    52. feature JBD2 3581 (0.004127)
    53. feature I2C 4591 (0.004125)
    54. feature INFINIBAND 903 (0.004118)
    55. feature SND_SOC_WM9081 3436 (0.004097)
    56. feature NOP_TRACER 8491 (0.004080)
    57. feature DRM_TTM 2377 (0.003717)
    58. feature REGMAP_I2C 1614 (0.003689)
    59. feature CPUMASK_OFFSTACK 8257 (0.003660)
    60. feature CRYPTO_PCRYPT 6580 (0.003591)
    61. feature FUNCTION_TRACER 1146 (0.003570)
    62. feature DRM_VM 2395 (0.003538)
    63. feature MAC80211_LEDS 7996 (0.003527)
    64. feature SND_SOC_HDMI_CODEC 2427 (0.003502)
    65. feature ACPI_BUTTON 5964 (0.003446)
    66. feature DVB_LGDT330X 7951 (0.003376)
    67. feature MEDIA_USB_SUPPORT 2494 (0.003333)
    68. feature SND_SOC_ARIZONA 7804 (0.003322)
    69. feature EFI_VARS 2290 (0.003298)
    70. feature VIDEOMODE_HELPERS 2753 (0.003273)
    71. feature SCSI_SAS_ATTRS 9074 (0.003232)
    72. feature SCSI_LPFC 4774 (0.003231)
    73. feature LOCKDEP 6832 (0.003201)
    74. feature VGA_CONSOLE 2080 (0.003182)
    75. feature INPUT_IMS_PCU 4924 (0.003166)
    76. feature MODVERSIONS 8428 (0.003162)
    77. feature GLOB 1613 (0.003154)
    78. feature COREDUMP 6451 (0.003091)
    79. feature NETDEV_NOTIFIER_ERROR_INJECT 5295 (0.003034)
    80. feature SND_SOC_WM1250_EV1 3449 (0.003026)
    81. feature INET6_XFRM_MODE_TRANSPORT 3354 (0.002901)
    82. feature INET6_TUNNEL 3347 (0.002856)
    83. feature MEDIA_TUNER_MT2063 2544 (0.002780)
    84. feature ADE7854_I2C 1218 (0.002724)
    85. feature BE2ISCSI 791 (0.002714)
    86. feature HID 5196 (0.002690)
    87. feature USB_NET_AX8817X 6842 (0.002632)
    88. feature V4L2_FWNODE 4105 (0.002582)
    89. feature CS5535_MFGPT 4519 (0.002543)
    90. feature MCP4531 1813 (0.002482)
    91. feature SND_SOC_CS4271_SPI 4902 (0.002464)
    92. feature SCSI_QLOGIC_1280 4736 (0.002460)
    93. feature USB_SERIAL_KOBIL_SCT 3948 (0.002440)
    94. feature VIDEO_WM8775 7918 (0.002327)
    95. feature DEBUG_PAGE_REF 4673 (0.002319)
    96. feature BLK_DEV_IO_TRACE 65 (0.002276)
    97. feature USB_STORAGE_ONETOUCH 632 (0.002262)
    98. feature MEDIA_TUNER_FC0013 2632 (0.002215)
    99. feature MDIO 7168 (0.002189)
    100. feature SIGMATEL_FIR 4785 (0.002176)
    Prediction score (MAE): 2.19
    Prediction score (MSE): 9797333.87
    Prediction score (R2): 0.93
    Prediction score (MRE): 7.72



```python
#re[56589]

```


```python
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.feature_selection import SelectFromModel

#model = SelectFromModel(clf, prefit=True)
#tuxdata_reduced = model.transform(tuxdata.drop(columns=size_methods))
#tuxdata_reduced.shape, tuxdata.shape

```


```python
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.svm import LinearSVR
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV, RFE


#alphas=[0.1, 1.0, 10.0, 100.0, 500.0, 750.0, 1000.0, 2000.0, 2500.0, 3000.0, 5000.0, 10000.0]
#selector = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 500.0, 750.0, 1000.0, 2000.0, 2500.0, 3000.0, 5000.0, 10000.0]) # LassoCV(tol = 0.001) # RidgeCV(alphas=[2000.0])  # 
#lass = selector #SelectFromModel(selector) #  RFECV(estimator=selector, step=1, scoring='neg_mean_squared_error') # 
#lass = RFE(estimator=selector, step=1)
#lass.fit(X_train, y_train)
#tuxdata_reduced_lass = lass.transform(tuxdata.drop(columns=size_methods))
#tuxdata_reduced_lass.shape, tuxdata.shape  
#lass.alpha_ 


```


```python
#from sklearn.decomposition import PCA

#pca = PCA(n_components=100)
#pca.fit(X_train, y_train)

#tuxdata_reduced_pca = pca.transform(tuxdata.drop(columns=size_methods))
#tuxdata_reduced_pca.shape, tuxdata.shape  

#pca.components_.shape

#plt.matshow(pca.components_, cmap='viridis')
#plt.yticks([0, 1], ["First component", "Second component"])
#plt.colorbar()
#plt.xticks(range(len(X_train.columns)),
#           X_train.columns, rotation=60, ha='left')
#plt.xlabel("Feature")
#plt.ylabel("Principal components")
```


```python
ft_vals = ['y', 'n'] 
tri_state_values = ['y', 'n', 'm']
all(x in tri_state_values for x in ft_vals)
```




    True




```python
#for tux1 in tuxdata:
#    ft1 = tuxdata[tux1]
#    for tux2 in tuxdata:
#        if (tux1 != tux2):
#            if (ft1.all() == tuxdata[tux2].all()):
#                print ("feature " + str(tux1) + " always have the same values than " + str(tux2))
            
    
```


```python
#provisoire = pd.read_csv(open('provisoire.csv', "r"))
```


```python
#provisoire[['cid','CC_OPTIMIZE_FOR_SIZE']]
```


```python
#rawtuxdata.columns[6015] #Columns (1150,6015,6026,7676,7726)
```


```python
#size_methods = ["vmlinux", "GZIP-bzImage", "GZIP-vmlinux", "GZIP", "BZIP2-bzImage", 
#              "BZIP2-vmlinux", "BZIP2", "LZMA-bzImage", "LZMA-vmlinux", "LZMA", "XZ-bzImage", "XZ-vmlinux", "XZ", 
#              "LZO-bzImage", "LZO-vmlinux", "LZO", "LZ4-bzImage", "LZ4-vmlinux", "LZ4"]
#size_methods_without_soi
```


```python
#import h2o
#from h2o.automl import H2OAutoML
#h2o.init()
#df = h2o.import_file(TUXML_CSV_FILENAME)
```


```python
#df.describe()
```


```python
#splits = df.split_frame(ratios = [0.8], seed = 1)
#train = splits[0]
#test = splits[1]
```


```python
#y = size_of_interest
#aml = H2OAutoML(max_runtime_secs = 36000, seed = 1, project_name = "tuxlearning")
#aml.train(y = y, training_frame = train, leaderboard_frame = test)
```


```python
#aml.leaderboard.head()
```


```python
#pred = aml.predict(test)
#pred.head()
```


```python
#perf = aml.leader.model_performance(test)
#perf
```


```python
#h2o.shutdown()
```


```python
#import category_encoders as ce

#colmatters = list(tuxdata.columns)
#for s in size_methods:
#    colmatters.remove(s)
    
# colmatters.remove(size_methods)
#encoder = ce.OneHotEncoder(cols=colmatters) #cols=tuxdata.drop(columns=size_methods).columns

#o_sizes = tuxdata[size_methods]
#encoder.fit(tuxdata.drop(columns=size_methods))
#tuxdata_dummy2 = encoder.transform(tuxdata.drop(columns=size_methods))
#tuxdata_dummy2[size_methods] = o_sizes
```


```python
#rawtuxdata[rawtuxdata['vmlinux'] == 1168072]#['MODULES']
```


```python
#tuxdata_dummy2.shape, tuxdata.shape
```


```python
#rawtuxdata[(rawtuxdata['MODULES'] == 'y')]['vmlinux'].describe(), rawtuxdata[(rawtuxdata['MODULES'] == 'n')]['vmlinux'].describe()
#rawtuxdata[(rawtuxdata['UBSAN_SANITIZE_ALL'] == 'y')]
# [['cid', 'CC_OPTIMIZE_FOR_SIZE', 'DEBUG_INFO_DWARF4', 'KASAN', 'UBSAN_ALIGNMENT', 'X86_NEED_RELOCS', 'RANDOMIZE_BASE', 'GCOV_PROFILE_ALL', 'UBSAN_SANITIZE_ALL', 'DEBUG_INFO', 'MODULES', 'DEBUG_INFO_REDUCED', 'DEBUG_INFO_SPLIT']]
```


```python
rawtuxdata.info(memory_usage='deep')
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2489 entries, 0 to 2499
    Columns: 9230 entries, OPENVSWITCH to LZ4
    dtypes: int64(9230)
    memory usage: 175.3 MB



```python
rawtuxdata['vmlinux'].sort_values()
```




    798      10889536
    2037     10892608
    2267     10962488
    778      11116560
    2080     11193720
    390      11534824
    935      11614640
    486      11702648
    717      11738880
    2265     11761680
    523      11924688
    1448     12021080
    2096     12054584
    2269     12119576
    2014     12373376
    1139     12376544
    1587     12437072
    1371     12644728
    1160     12995432
    1741     13053992
    2296     13087256
    2365     13105144
    1488     13226152
    1069     13335744
    2352     13437512
    681      13461864
    1976     13485192
    1597     13511520
    2201     13608016
    2076     13664568
              ...    
    2491     69128056
    1858     69264288
    604      69480528
    137      70378616
    52       70712600
    1483     71107896
    2059     71776032
    1506     72795712
    1624     73126000
    2478     73279672
    181      73555192
    2291     74025704
    1340     74077552
    2468     74098352
    1025     74712680
    27       74917072
    1663     75693488
    2098     76351856
    1969     79124240
    2123     81215616
    1994     86371944
    1992     87491520
    588      88796896
    1157     89448144
    125      89971128
    1659     93768688
    297      95478144
    939     101323320
    373     107293136
    2190    129896344
    Name: vmlinux, Length: 2489, dtype: int64


