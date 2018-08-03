

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


    Original size (#configs/#options) of the dataset (30637, 12798)
    Number of options with only one value (eg always y): (2903, 1)
    Non tri-state value options (eg string or integer or hybrid values): (173, 1) 
    Predictor variables: 9722



```python
'X86_64' in ftuniques, 'DEBUG_INFO' in ftuniques, 'GCOV_PROFILE_ALL' in ftuniques, 'KASAN' in ftuniques, 'UBSAN_SANITIZE_ALL' in ftuniques, 'RELOCATABLE' in ftuniques, 'XFS_DEBUG' in ftuniques, 'AIC7XXX_BUILD_FIRMWARE' in ftuniques, 'AIC79XX_BUILD_FIRMWARE' in ftuniques, 'WANXL_BUILD_FIRMWARE' in ftuniques
```




    (False, False, False, False, False, False, False, False, False, False)




```python
if 'RELOCATABLE' in rawtuxdata.columns:
    print(rawtuxdata.query("RELOCATABLE == 'y'")[['cid', 'RELOCATABLE']])
```

             cid RELOCATABLE
    0      62000           y
    1      62001           y
    2      62002           y
    3      62003           y
    6      62006           y
    7      62007           y
    8      62008           y
    9      62009           y
    10     62010           y
    12     62012           y
    13     62013           y
    14     62014           y
    15     62015           y
    17     62017           y
    18     62018           y
    19     62019           y
    20     62020           y
    21     62021           y
    23     62023           y
    25     62025           y
    27     62027           y
    32     62032           y
    36     62036           y
    38     62038           y
    39     62039           y
    40     62040           y
    42     62042           y
    44     62044           y
    45     62045           y
    46     62046           y
    ...      ...         ...
    30355  92518           y
    30366  92529           y
    30371  92534           y
    30382  92545           y
    30400  92563           y
    30402  92565           y
    30418  92581           y
    30423  92586           y
    30448  92611           y
    30450  92613           y
    30451  92614           y
    30459  92622           y
    30470  92633           y
    30480  92643           y
    30485  92648           y
    30504  92667           y
    30506  92669           y
    30520  92683           y
    30552  92715           y
    30556  92719           y
    30558  92721           y
    30559  92722           y
    30560  92723           y
    30583  92746           y
    30600  92763           y
    30605  92768           y
    30614  92777           y
    30624  92787           y
    30627  92790           y
    30635  92798           y
    
    [11372 rows x 2 columns]



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
      <th>SND_SOC_INTEL_SKL_NAU88L25_SSM4567_MACH</th>
      <th>APDS9960</th>
      <th>ARCH_SUPPORTS_INT128</th>
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
      <th>12</th>
      <td>62012</td>
      <td>2018-06-27 08:53:58</td>
      <td>650.557</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>19</th>
      <td>62019</td>
      <td>2018-06-27 08:55:15</td>
      <td>803.271</td>
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
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
    </tr>
    <tr>
      <th>25</th>
      <td>62025</td>
      <td>2018-06-27 08:56:52</td>
      <td>204.002</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>69</th>
      <td>62069</td>
      <td>2018-06-27 09:04:39</td>
      <td>616.790</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>70</th>
      <td>62070</td>
      <td>2018-06-27 09:04:38</td>
      <td>522.449</td>
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
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
    </tr>
    <tr>
      <th>93</th>
      <td>62093</td>
      <td>2018-06-27 09:09:30</td>
      <td>604.755</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>146</th>
      <td>62146</td>
      <td>2018-06-27 09:18:33</td>
      <td>625.524</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>169</th>
      <td>62169</td>
      <td>2018-06-27 09:21:46</td>
      <td>461.916</td>
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
      <td>n</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>187</th>
      <td>62187</td>
      <td>2018-06-27 09:25:04</td>
      <td>362.615</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
    </tr>
    <tr>
      <th>214</th>
      <td>62214</td>
      <td>2018-06-27 09:30:04</td>
      <td>573.729</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>218</th>
      <td>62218</td>
      <td>2018-06-27 09:30:49</td>
      <td>432.964</td>
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
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>257</th>
      <td>62257</td>
      <td>2018-06-27 15:08:09</td>
      <td>231.720</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>263</th>
      <td>62263</td>
      <td>2018-06-27 15:09:28</td>
      <td>277.047</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>n</td>
      <td>m</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
      <td>n</td>
    </tr>
    <tr>
      <th>268</th>
      <td>62268</td>
      <td>2018-06-27 15:09:55</td>
      <td>322.275</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>272</th>
      <td>62272</td>
      <td>2018-06-27 15:10:28</td>
      <td>351.339</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>276</th>
      <td>62276</td>
      <td>2018-06-27 15:11:52</td>
      <td>417.822</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>n</td>
      <td>m</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>299</th>
      <td>62299</td>
      <td>2018-06-27 15:13:52</td>
      <td>534.526</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>n</td>
      <td>m</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>m</td>
      <td>y</td>
    </tr>
    <tr>
      <th>313</th>
      <td>62313</td>
      <td>2018-06-27 15:15:32</td>
      <td>557.893</td>
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
      <td>m</td>
      <td>n</td>
      <td>m</td>
      <td>y</td>
    </tr>
    <tr>
      <th>326</th>
      <td>62326</td>
      <td>2018-06-27 15:17:47</td>
      <td>438.370</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>345</th>
      <td>62345</td>
      <td>2018-06-27 15:20:19</td>
      <td>290.777</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>n</td>
      <td>m</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>m</td>
      <td>n</td>
    </tr>
    <tr>
      <th>365</th>
      <td>62365</td>
      <td>2018-06-27 15:22:31</td>
      <td>445.306</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>385</th>
      <td>62385</td>
      <td>2018-06-27 15:26:04</td>
      <td>272.338</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>389</th>
      <td>62389</td>
      <td>2018-06-27 15:27:13</td>
      <td>390.972</td>
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
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>447</th>
      <td>62447</td>
      <td>2018-06-27 15:35:27</td>
      <td>317.198</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>501</th>
      <td>62501</td>
      <td>2018-06-27 15:42:22</td>
      <td>556.812</td>
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
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
    </tr>
    <tr>
      <th>533</th>
      <td>62533</td>
      <td>2018-06-27 15:48:14</td>
      <td>385.966</td>
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
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
    </tr>
    <tr>
      <th>561</th>
      <td>62561</td>
      <td>2018-06-27 15:51:32</td>
      <td>396.008</td>
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
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
    </tr>
    <tr>
      <th>592</th>
      <td>62592</td>
      <td>2018-06-27 15:56:14</td>
      <td>221.885</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>602</th>
      <td>62602</td>
      <td>2018-06-27 15:57:24</td>
      <td>303.841</td>
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
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>632</th>
      <td>62632</td>
      <td>2018-06-27 16:01:41</td>
      <td>295.292</td>
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
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>m</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24794</th>
      <td>86957</td>
      <td>2018-07-27 17:07:14</td>
      <td>363.165</td>
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
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>24881</th>
      <td>87044</td>
      <td>2018-07-27 17:45:18</td>
      <td>1177.750</td>
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
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
    </tr>
    <tr>
      <th>24911</th>
      <td>87074</td>
      <td>2018-07-27 18:09:20</td>
      <td>377.185</td>
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
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>25177</th>
      <td>87340</td>
      <td>2018-07-28 06:58:10</td>
      <td>4473.540</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
    </tr>
    <tr>
      <th>25211</th>
      <td>87374</td>
      <td>2018-07-28 08:54:17</td>
      <td>2256.470</td>
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
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>25351</th>
      <td>87514</td>
      <td>2018-07-28 16:17:02</td>
      <td>1864.600</td>
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
      <th>25403</th>
      <td>87566</td>
      <td>2018-07-28 19:52:05</td>
      <td>3040.100</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>25487</th>
      <td>87650</td>
      <td>2018-07-29 03:08:54</td>
      <td>2842.170</td>
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
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>25977</th>
      <td>88140</td>
      <td>2018-07-30 11:19:22</td>
      <td>474.292</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>25989</th>
      <td>88152</td>
      <td>2018-07-30 11:22:40</td>
      <td>639.324</td>
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
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>26089</th>
      <td>88252</td>
      <td>2018-07-30 11:47:14</td>
      <td>373.653</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>26585</th>
      <td>88748</td>
      <td>2018-07-30 13:55:54</td>
      <td>713.699</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>26654</th>
      <td>88817</td>
      <td>2018-07-30 14:14:02</td>
      <td>520.940</td>
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
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>26746</th>
      <td>88909</td>
      <td>2018-07-30 14:39:02</td>
      <td>458.704</td>
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
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>27414</th>
      <td>89577</td>
      <td>2018-07-30 18:42:28</td>
      <td>792.406</td>
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
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>27455</th>
      <td>89618</td>
      <td>2018-07-30 18:57:54</td>
      <td>328.483</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
      <td>n</td>
    </tr>
    <tr>
      <th>27538</th>
      <td>89701</td>
      <td>2018-07-30 19:34:46</td>
      <td>811.173</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>27594</th>
      <td>89757</td>
      <td>2018-07-30 19:58:09</td>
      <td>2790.040</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>27691</th>
      <td>89854</td>
      <td>2018-07-30 20:46:31</td>
      <td>701.857</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
    </tr>
    <tr>
      <th>28343</th>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>28547</th>
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
      <td>y</td>
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
      <th>28812</th>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>28891</th>
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
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>29026</th>
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
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>29128</th>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
    </tr>
    <tr>
      <th>29470</th>
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
      <td>y</td>
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
      <th>29579</th>
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
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>29699</th>
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
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>29909</th>
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
      <td>y</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>29957</th>
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
      <td>y</td>
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
<p>1507 rows × 9895 columns</p>
</div>




```python
#rawtuxdata[rawtuxdata['vmlinux'] == 1168072]['MODULES']
rawtuxdata.query("vmlinux == 1168072")['MODULES'] #tiny config for X86_32
```




    8604     n
    8608     n
    8617     n
    8618     n
    24554    n
    Name: MODULES, dtype: object




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




    0      55559344
    1      37436552
    2      68465904
    3      87309464
    4      35289888
    5      25067464
    6      33254424
    7      33917232
    8      34379048
    9     168606904
    10    304245920
    11     24167768
    12           -1
    13     53676608
    14     67717688
    15     22804840
    16     47719480
    17     13585048
    18     74078032
    19           -1
    Name: vmlinux, dtype: int64




```python
rawtuxdata.shape, rawtuxdata.query("vmlinux != -1").shape
```




    ((30637, 9895), (29130, 9895))




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
    count     30637
    unique        2
    top           y
    freq      30607
    Name: X86_64, dtype: object
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 30607 entries, 0 to 30636
    Columns: 9895 entries, cid to NETWORK_FILESYSTEMS
    dtypes: float64(1), int64(153), object(9741)
    memory usage: 18.3 GB



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

    30607 before the removal of some entries (those with same configurations)
    30589 after the removal of some entries (those with same configurations)
    29098 after the removal of configurations that do NOT compile



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
      <th>SND_SOC_INTEL_SKL_NAU88L25_SSM4567_MACH</th>
      <th>APDS9960</th>
      <th>ARCH_SUPPORTS_INT128</th>
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
<p>0 rows × 9895 columns</p>
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
      <th>SND_SOC_INTEL_SKL_NAU88L25_SSM4567_MACH</th>
      <th>APDS9960</th>
      <th>ARCH_SUPPORTS_INT128</th>
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
      <th>12299</th>
      <td>74458</td>
      <td>2018-07-05 16:07:21</td>
      <td>28.7108</td>
      <td>7317008</td>
      <td>646608</td>
      <td>2733176</td>
      <td>501222</td>
      <td>4722128</td>
      <td>6808144</td>
      <td>458568</td>
      <td>...</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 9895 columns</p>
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





    count    2.909800e+04
    mean     5.146904e+07
    std      6.787058e+07
    min      7.317008e+06
    25%      2.410287e+07
    50%      3.378448e+07
    75%      5.126702e+07
    max      1.693674e+09
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
      <th>SND_SOC_INTEL_SKL_NAU88L25_SSM4567_MACH</th>
      <th>APDS9960</th>
      <th>ARCH_SUPPORTS_INT128</th>
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
<p>0 rows × 9895 columns</p>
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
    #T_7c84e246_9574_11e8_970b_525400123456row0_col0 {
            color:  black;
        }    #T_7c84e246_9574_11e8_970b_525400123456row0_col1 {
            color:  green;
        }    #T_7c84e246_9574_11e8_970b_525400123456row0_col2 {
            color:  green;
        }    #T_7c84e246_9574_11e8_970b_525400123456row0_col3 {
            color:  green;
        }    #T_7c84e246_9574_11e8_970b_525400123456row0_col4 {
            color:  red;
        }    #T_7c84e246_9574_11e8_970b_525400123456row0_col5 {
            color:  red;
        }    #T_7c84e246_9574_11e8_970b_525400123456row1_col0 {
            color:  red;
        }    #T_7c84e246_9574_11e8_970b_525400123456row1_col1 {
            color:  black;
        }    #T_7c84e246_9574_11e8_970b_525400123456row1_col2 {
            color:  green;
        }    #T_7c84e246_9574_11e8_970b_525400123456row1_col3 {
            color:  green;
        }    #T_7c84e246_9574_11e8_970b_525400123456row1_col4 {
            color:  red;
        }    #T_7c84e246_9574_11e8_970b_525400123456row1_col5 {
            color:  red;
        }    #T_7c84e246_9574_11e8_970b_525400123456row2_col0 {
            color:  red;
        }    #T_7c84e246_9574_11e8_970b_525400123456row2_col1 {
            color:  red;
        }    #T_7c84e246_9574_11e8_970b_525400123456row2_col2 {
            color:  black;
        }    #T_7c84e246_9574_11e8_970b_525400123456row2_col3 {
            color:  green;
        }    #T_7c84e246_9574_11e8_970b_525400123456row2_col4 {
            color:  red;
        }    #T_7c84e246_9574_11e8_970b_525400123456row2_col5 {
            color:  red;
        }    #T_7c84e246_9574_11e8_970b_525400123456row3_col0 {
            color:  red;
        }    #T_7c84e246_9574_11e8_970b_525400123456row3_col1 {
            color:  red;
        }    #T_7c84e246_9574_11e8_970b_525400123456row3_col2 {
            color:  red;
        }    #T_7c84e246_9574_11e8_970b_525400123456row3_col3 {
            color:  black;
        }    #T_7c84e246_9574_11e8_970b_525400123456row3_col4 {
            color:  red;
        }    #T_7c84e246_9574_11e8_970b_525400123456row3_col5 {
            color:  red;
        }    #T_7c84e246_9574_11e8_970b_525400123456row4_col0 {
            color:  green;
        }    #T_7c84e246_9574_11e8_970b_525400123456row4_col1 {
            color:  green;
        }    #T_7c84e246_9574_11e8_970b_525400123456row4_col2 {
            color:  green;
        }    #T_7c84e246_9574_11e8_970b_525400123456row4_col3 {
            color:  green;
        }    #T_7c84e246_9574_11e8_970b_525400123456row4_col4 {
            color:  black;
        }    #T_7c84e246_9574_11e8_970b_525400123456row4_col5 {
            color:  red;
        }    #T_7c84e246_9574_11e8_970b_525400123456row5_col0 {
            color:  green;
        }    #T_7c84e246_9574_11e8_970b_525400123456row5_col1 {
            color:  green;
        }    #T_7c84e246_9574_11e8_970b_525400123456row5_col2 {
            color:  green;
        }    #T_7c84e246_9574_11e8_970b_525400123456row5_col3 {
            color:  green;
        }    #T_7c84e246_9574_11e8_970b_525400123456row5_col4 {
            color:  green;
        }    #T_7c84e246_9574_11e8_970b_525400123456row5_col5 {
            color:  black;
        }</style>  
<table id="T_7c84e246_9574_11e8_970b_525400123456" ><caption>Difference (average in percentage) per compression methods</caption> 
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
        <th id="T_7c84e246_9574_11e8_970b_525400123456level0_row0" class="row_heading level0 row0" >GZIPo</th> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row0_col0" class="data row0 col0" >0</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row0_col1" class="data row0 col1" >3.00409</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row0_col2" class="data row0 col2" >24.1075</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row0_col3" class="data row0 col3" >37.5807</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row0_col4" class="data row0 col4" >-9.84647</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row0_col5" class="data row0 col5" >-37771.6</td> 
    </tr>    <tr> 
        <th id="T_7c84e246_9574_11e8_970b_525400123456level0_row1" class="row_heading level0 row1" >BZIP2o</th> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row1_col0" class="data row1 col0" >-2.85938</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row1_col1" class="data row1 col1" >0</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row1_col2" class="data row1 col2" >20.5629</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row1_col3" class="data row1 col3" >33.6321</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row1_col4" class="data row1 col4" >-12.4552</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row1_col5" class="data row1 col5" >-36319.8</td> 
    </tr>    <tr> 
        <th id="T_7c84e246_9574_11e8_970b_525400123456level0_row2" class="row_heading level0 row2" >LZMAo</th> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row2_col0" class="data row2 col0" >-19.3674</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row2_col1" class="data row2 col1" >-16.9444</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row2_col2" class="data row2 col2" >0</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row2_col3" class="data row2 col3" >10.8618</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row2_col4" class="data row2 col4" >-27.2786</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row2_col5" class="data row2 col5" >-29692.8</td> 
    </tr>    <tr> 
        <th id="T_7c84e246_9574_11e8_970b_525400123456level0_row3" class="row_heading level0 row3" >XZo</th> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row3_col0" class="data row3 col0" >-27.0015</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row3_col1" class="data row3 col1" >-24.8181</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row3_col2" class="data row3 col2" >-9.46493</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row3_col3" class="data row3 col3" >0</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row3_col4" class="data row3 col4" >-34.1699</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row3_col5" class="data row3 col5" >-27520.5</td> 
    </tr>    <tr> 
        <th id="T_7c84e246_9574_11e8_970b_525400123456level0_row4" class="row_heading level0 row4" >LZOo</th> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row4_col0" class="data row4 col0" >10.9803</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row4_col1" class="data row4 col1" >14.2732</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row4_col2" class="data row4 col2" >37.7969</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row4_col3" class="data row4 col3" >52.7391</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row4_col4" class="data row4 col4" >0</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row4_col5" class="data row4 col5" >-41972.2</td> 
    </tr>    <tr> 
        <th id="T_7c84e246_9574_11e8_970b_525400123456level0_row5" class="row_heading level0 row5" >LZ4o</th> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row5_col0" class="data row5 col0" >18.834</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row5_col1" class="data row5 col1" >22.3366</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row5_col2" class="data row5 col2" >47.584</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row5_col3" class="data row5 col3" >63.5921</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row5_col4" class="data row5 col4" >7.04909</td> 
        <td id="T_7c84e246_9574_11e8_970b_525400123456row5_col5" class="data row5 col5" >0</td> 
    </tr></tbody> 
</table> 




```python
compareCompress("-bzImage").style.set_caption('Difference (average in percentage) per compression methods, bzImage').applymap(color_negative_positive)

```




<style  type="text/css" >
    #T_7c84e247_9574_11e8_970b_525400123456row0_col0 {
            color:  black;
        }    #T_7c84e247_9574_11e8_970b_525400123456row0_col1 {
            color:  red;
        }    #T_7c84e247_9574_11e8_970b_525400123456row0_col2 {
            color:  green;
        }    #T_7c84e247_9574_11e8_970b_525400123456row0_col3 {
            color:  green;
        }    #T_7c84e247_9574_11e8_970b_525400123456row0_col4 {
            color:  red;
        }    #T_7c84e247_9574_11e8_970b_525400123456row0_col5 {
            color:  red;
        }    #T_7c84e247_9574_11e8_970b_525400123456row1_col0 {
            color:  green;
        }    #T_7c84e247_9574_11e8_970b_525400123456row1_col1 {
            color:  black;
        }    #T_7c84e247_9574_11e8_970b_525400123456row1_col2 {
            color:  green;
        }    #T_7c84e247_9574_11e8_970b_525400123456row1_col3 {
            color:  green;
        }    #T_7c84e247_9574_11e8_970b_525400123456row1_col4 {
            color:  green;
        }    #T_7c84e247_9574_11e8_970b_525400123456row1_col5 {
            color:  green;
        }    #T_7c84e247_9574_11e8_970b_525400123456row2_col0 {
            color:  red;
        }    #T_7c84e247_9574_11e8_970b_525400123456row2_col1 {
            color:  red;
        }    #T_7c84e247_9574_11e8_970b_525400123456row2_col2 {
            color:  black;
        }    #T_7c84e247_9574_11e8_970b_525400123456row2_col3 {
            color:  green;
        }    #T_7c84e247_9574_11e8_970b_525400123456row2_col4 {
            color:  red;
        }    #T_7c84e247_9574_11e8_970b_525400123456row2_col5 {
            color:  red;
        }    #T_7c84e247_9574_11e8_970b_525400123456row3_col0 {
            color:  red;
        }    #T_7c84e247_9574_11e8_970b_525400123456row3_col1 {
            color:  red;
        }    #T_7c84e247_9574_11e8_970b_525400123456row3_col2 {
            color:  red;
        }    #T_7c84e247_9574_11e8_970b_525400123456row3_col3 {
            color:  black;
        }    #T_7c84e247_9574_11e8_970b_525400123456row3_col4 {
            color:  red;
        }    #T_7c84e247_9574_11e8_970b_525400123456row3_col5 {
            color:  red;
        }    #T_7c84e247_9574_11e8_970b_525400123456row4_col0 {
            color:  green;
        }    #T_7c84e247_9574_11e8_970b_525400123456row4_col1 {
            color:  red;
        }    #T_7c84e247_9574_11e8_970b_525400123456row4_col2 {
            color:  green;
        }    #T_7c84e247_9574_11e8_970b_525400123456row4_col3 {
            color:  green;
        }    #T_7c84e247_9574_11e8_970b_525400123456row4_col4 {
            color:  black;
        }    #T_7c84e247_9574_11e8_970b_525400123456row4_col5 {
            color:  red;
        }    #T_7c84e247_9574_11e8_970b_525400123456row5_col0 {
            color:  green;
        }    #T_7c84e247_9574_11e8_970b_525400123456row5_col1 {
            color:  red;
        }    #T_7c84e247_9574_11e8_970b_525400123456row5_col2 {
            color:  green;
        }    #T_7c84e247_9574_11e8_970b_525400123456row5_col3 {
            color:  green;
        }    #T_7c84e247_9574_11e8_970b_525400123456row5_col4 {
            color:  green;
        }    #T_7c84e247_9574_11e8_970b_525400123456row5_col5 {
            color:  black;
        }</style>  
<table id="T_7c84e247_9574_11e8_970b_525400123456" ><caption>Difference (average in percentage) per compression methods, bzImage</caption> 
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
        <th id="T_7c84e247_9574_11e8_970b_525400123456level0_row0" class="row_heading level0 row0" >GZIPo</th> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row0_col0" class="data row0 col0" >0</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row0_col1" class="data row0 col1" >-32.7688</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row0_col2" class="data row0 col2" >23.7047</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row0_col3" class="data row0 col3" >36.554</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row0_col4" class="data row0 col4" >-9.51477</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row0_col5" class="data row0 col5" >-15.4669</td> 
    </tr>    <tr> 
        <th id="T_7c84e247_9574_11e8_970b_525400123456level0_row1" class="row_heading level0 row1" >BZIP2o</th> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row1_col0" class="data row1 col0" >53.1827</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row1_col1" class="data row1 col1" >0</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row1_col2" class="data row1 col2" >88.9821</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row1_col3" class="data row1 col3" >108.159</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row1_col4" class="data row1 col4" >38.7021</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row1_col5" class="data row1 col5" >29.6217</td> 
    </tr>    <tr> 
        <th id="T_7c84e247_9574_11e8_970b_525400123456level0_row2" class="row_heading level0 row2" >LZMAo</th> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row2_col0" class="data row2 col0" >-19.1039</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row2_col1" class="data row2 col1" >-45.7586</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row2_col2" class="data row2 col2" >0</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row2_col3" class="data row2 col3" >10.3884</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row2_col4" class="data row2 col4" >-26.772</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row2_col5" class="data row2 col5" >-31.5763</td> 
    </tr>    <tr> 
        <th id="T_7c84e247_9574_11e8_970b_525400123456level0_row3" class="row_heading level0 row3" >XZo</th> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row3_col0" class="data row3 col0" >-26.4578</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row3_col1" class="data row3 col1" >-50.7791</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row3_col2" class="data row3 col2" >-9.09136</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row3_col3" class="data row3 col3" >0</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row3_col4" class="data row3 col4" >-33.4336</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row3_col5" class="data row3 col5" >-37.7988</td> 
    </tr>    <tr> 
        <th id="T_7c84e247_9574_11e8_970b_525400123456level0_row4" class="row_heading level0 row4" >LZOo</th> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row4_col0" class="data row4 col0" >10.5725</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row4_col1" class="data row4 col1" >-25.6061</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row4_col2" class="data row4 col2" >36.8458</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row4_col3" class="data row4 col3" >51.047</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row4_col4" class="data row4 col4" >0</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row4_col5" class="data row4 col5" >-6.59913</td> 
    </tr>    <tr> 
        <th id="T_7c84e247_9574_11e8_970b_525400123456level0_row5" class="row_heading level0 row5" >LZ4o</th> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row5_col0" class="data row5 col0" >18.4297</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row5_col1" class="data row5 col1" >-20.2875</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row5_col2" class="data row5 col2" >46.6039</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row5_col3" class="data row5 col3" >61.822</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row5_col4" class="data row5 col4" >7.07913</td> 
        <td id="T_7c84e247_9574_11e8_970b_525400123456row5_col5" class="data row5 col5" >0</td> 
    </tr></tbody> 
</table> 




```python
compareCompress("-vmlinux").style.set_caption('Difference (average in percentage) per compression methods, vmlinux').applymap(color_negative_positive)

```




<style  type="text/css" >
    #T_7c84e248_9574_11e8_970b_525400123456row0_col0 {
            color:  black;
        }    #T_7c84e248_9574_11e8_970b_525400123456row0_col1 {
            color:  red;
        }    #T_7c84e248_9574_11e8_970b_525400123456row0_col2 {
            color:  green;
        }    #T_7c84e248_9574_11e8_970b_525400123456row0_col3 {
            color:  green;
        }    #T_7c84e248_9574_11e8_970b_525400123456row0_col4 {
            color:  red;
        }    #T_7c84e248_9574_11e8_970b_525400123456row0_col5 {
            color:  red;
        }    #T_7c84e248_9574_11e8_970b_525400123456row1_col0 {
            color:  green;
        }    #T_7c84e248_9574_11e8_970b_525400123456row1_col1 {
            color:  black;
        }    #T_7c84e248_9574_11e8_970b_525400123456row1_col2 {
            color:  green;
        }    #T_7c84e248_9574_11e8_970b_525400123456row1_col3 {
            color:  green;
        }    #T_7c84e248_9574_11e8_970b_525400123456row1_col4 {
            color:  green;
        }    #T_7c84e248_9574_11e8_970b_525400123456row1_col5 {
            color:  green;
        }    #T_7c84e248_9574_11e8_970b_525400123456row2_col0 {
            color:  red;
        }    #T_7c84e248_9574_11e8_970b_525400123456row2_col1 {
            color:  red;
        }    #T_7c84e248_9574_11e8_970b_525400123456row2_col2 {
            color:  black;
        }    #T_7c84e248_9574_11e8_970b_525400123456row2_col3 {
            color:  green;
        }    #T_7c84e248_9574_11e8_970b_525400123456row2_col4 {
            color:  red;
        }    #T_7c84e248_9574_11e8_970b_525400123456row2_col5 {
            color:  red;
        }    #T_7c84e248_9574_11e8_970b_525400123456row3_col0 {
            color:  red;
        }    #T_7c84e248_9574_11e8_970b_525400123456row3_col1 {
            color:  red;
        }    #T_7c84e248_9574_11e8_970b_525400123456row3_col2 {
            color:  red;
        }    #T_7c84e248_9574_11e8_970b_525400123456row3_col3 {
            color:  black;
        }    #T_7c84e248_9574_11e8_970b_525400123456row3_col4 {
            color:  red;
        }    #T_7c84e248_9574_11e8_970b_525400123456row3_col5 {
            color:  red;
        }    #T_7c84e248_9574_11e8_970b_525400123456row4_col0 {
            color:  green;
        }    #T_7c84e248_9574_11e8_970b_525400123456row4_col1 {
            color:  red;
        }    #T_7c84e248_9574_11e8_970b_525400123456row4_col2 {
            color:  green;
        }    #T_7c84e248_9574_11e8_970b_525400123456row4_col3 {
            color:  green;
        }    #T_7c84e248_9574_11e8_970b_525400123456row4_col4 {
            color:  black;
        }    #T_7c84e248_9574_11e8_970b_525400123456row4_col5 {
            color:  red;
        }    #T_7c84e248_9574_11e8_970b_525400123456row5_col0 {
            color:  green;
        }    #T_7c84e248_9574_11e8_970b_525400123456row5_col1 {
            color:  red;
        }    #T_7c84e248_9574_11e8_970b_525400123456row5_col2 {
            color:  green;
        }    #T_7c84e248_9574_11e8_970b_525400123456row5_col3 {
            color:  green;
        }    #T_7c84e248_9574_11e8_970b_525400123456row5_col4 {
            color:  green;
        }    #T_7c84e248_9574_11e8_970b_525400123456row5_col5 {
            color:  black;
        }</style>  
<table id="T_7c84e248_9574_11e8_970b_525400123456" ><caption>Difference (average in percentage) per compression methods, vmlinux</caption> 
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
        <th id="T_7c84e248_9574_11e8_970b_525400123456level0_row0" class="row_heading level0 row0" >GZIPo</th> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row0_col0" class="data row0 col0" >0</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row0_col1" class="data row0 col1" >-27.4404</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row0_col2" class="data row0 col2" >17.9974</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row0_col3" class="data row0 col3" >27.0732</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row0_col4" class="data row0 col4" >-7.67451</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row0_col5" class="data row0 col5" >-12.6286</td> 
    </tr>    <tr> 
        <th id="T_7c84e248_9574_11e8_970b_525400123456level0_row1" class="row_heading level0 row1" >BZIP2o</th> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row1_col0" class="data row1 col0" >39.6672</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row1_col1" class="data row1 col1" >0</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row1_col2" class="data row1 col2" >64.2376</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row1_col3" class="data row1 col3" >76.5538</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row1_col4" class="data row1 col4" >29.1064</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row1_col5" class="data row1 col5" >22.2699</td> 
    </tr>    <tr> 
        <th id="T_7c84e248_9574_11e8_970b_525400123456level0_row2" class="row_heading level0 row2" >LZMAo</th> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row2_col0" class="data row2 col0" >-15.147</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row2_col1" class="data row2 col1" >-38.6417</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row2_col2" class="data row2 col2" >0</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row2_col3" class="data row2 col3" >7.65599</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row2_col4" class="data row2 col4" >-21.613</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row2_col5" class="data row2 col5" >-25.7954</td> 
    </tr>    <tr> 
        <th id="T_7c84e248_9574_11e8_970b_525400123456level0_row3" class="row_heading level0 row3" >XZo</th> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row3_col0" class="data row3 col0" >-20.988</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row3_col1" class="data row3 col1" >-42.9628</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row3_col2" class="data row3 col2" >-6.91522</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row3_col3" class="data row3 col3" >0</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row3_col4" class="data row3 col4" >-27.0016</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row3_col5" class="data row3 col5" >-30.889</td> 
    </tr>    <tr> 
        <th id="T_7c84e248_9574_11e8_970b_525400123456level0_row4" class="row_heading level0 row4" >LZOo</th> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row4_col0" class="data row4 col0" >8.36892</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row4_col1" class="data row4 col1" >-21.2692</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row4_col2" class="data row4 col2" >27.957</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row4_col3" class="data row4 col3" >37.8114</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row4_col4" class="data row4 col4" >0</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row4_col5" class="data row4 col5" >-5.38986</td> 
    </tr>    <tr> 
        <th id="T_7c84e248_9574_11e8_970b_525400123456level0_row5" class="row_heading level0 row5" >LZ4o</th> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row5_col0" class="data row5 col0" >14.5917</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row5_col1" class="data row5 col1" >-16.683</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row5_col2" class="data row5 col2" >35.356</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row5_col3" class="data row5 col3" >45.7952</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row5_col4" class="data row5 col4" >5.71283</td> 
        <td id="T_7c84e248_9574_11e8_970b_525400123456row5_col5" class="data row5 col5" >0</td> 
    </tr></tbody> 
</table> 




```python
cm = sns.light_palette("green", as_cmap=True)
pd.DataFrame.corr(rawtuxdata[size_methods]).style.set_caption('Correlations between size measures').background_gradient(cmap=cm)

```




<style  type="text/css" >
    #T_7c84e249_9574_11e8_970b_525400123456row0_col0 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col1 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col2 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col3 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col4 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col5 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col6 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col7 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col8 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col9 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col10 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col11 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col12 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col13 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col14 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col15 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col16 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col17 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row0_col18 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col0 {
            background-color:  #defbde;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col1 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col2 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col3 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col4 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col5 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col6 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col7 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col8 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col9 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col10 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col11 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col12 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col13 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col14 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col15 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col16 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col17 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row1_col18 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col0 {
            background-color:  #defbde;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col1 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col2 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col3 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col4 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col5 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col6 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col7 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col8 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col9 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col10 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col11 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col12 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col13 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col14 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col15 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col16 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col17 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row2_col18 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col0 {
            background-color:  #defbde;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col1 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col2 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col3 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col4 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col5 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col6 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col7 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col8 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col9 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col10 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col11 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col12 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col13 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col14 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col15 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col16 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col17 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row3_col18 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col0 {
            background-color:  #d8f8d8;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col1 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col2 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col3 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col4 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col5 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col6 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col7 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col8 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col9 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col10 {
            background-color:  #048204;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col11 {
            background-color:  #048204;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col12 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col13 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col14 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col15 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col16 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col17 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row4_col18 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col0 {
            background-color:  #d8f8d8;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col1 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col2 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col3 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col4 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col5 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col6 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col7 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col8 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col9 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col10 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col11 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col12 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col13 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col14 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col15 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col16 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col17 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row5_col18 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col0 {
            background-color:  #d8f8d8;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col1 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col2 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col3 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col4 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col5 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col6 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col7 {
            background-color:  #028102;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col8 {
            background-color:  #028102;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col9 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col10 {
            background-color:  #048204;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col11 {
            background-color:  #048204;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col12 {
            background-color:  #048204;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col13 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col14 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col15 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col16 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col17 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row6_col18 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col0 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col1 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col2 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col3 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col4 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col5 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col6 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col7 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col8 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col9 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col10 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col11 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col12 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col13 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col14 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col15 {
            background-color:  #048204;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col16 {
            background-color:  #068306;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col17 {
            background-color:  #068306;
        }    #T_7c84e249_9574_11e8_970b_525400123456row7_col18 {
            background-color:  #068306;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col0 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col1 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col2 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col3 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col4 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col5 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col6 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col7 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col8 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col9 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col10 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col11 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col12 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col13 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col14 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col15 {
            background-color:  #048204;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col16 {
            background-color:  #068306;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col17 {
            background-color:  #068306;
        }    #T_7c84e249_9574_11e8_970b_525400123456row8_col18 {
            background-color:  #068306;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col0 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col1 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col2 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col3 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col4 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col5 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col6 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col7 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col8 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col9 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col10 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col11 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col12 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col13 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col14 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col15 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col16 {
            background-color:  #078407;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col17 {
            background-color:  #078407;
        }    #T_7c84e249_9574_11e8_970b_525400123456row9_col18 {
            background-color:  #068306;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col0 {
            background-color:  #e4fee4;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col1 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col2 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col3 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col4 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col5 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col6 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col7 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col8 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col9 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col10 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col11 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col12 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col13 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col14 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col15 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col16 {
            background-color:  #088408;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col17 {
            background-color:  #088408;
        }    #T_7c84e249_9574_11e8_970b_525400123456row10_col18 {
            background-color:  #088408;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col0 {
            background-color:  #e4fee4;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col1 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col2 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col3 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col4 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col5 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col6 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col7 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col8 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col9 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col10 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col11 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col12 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col13 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col14 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col15 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col16 {
            background-color:  #088408;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col17 {
            background-color:  #088408;
        }    #T_7c84e249_9574_11e8_970b_525400123456row11_col18 {
            background-color:  #088408;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col0 {
            background-color:  #e5ffe5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col1 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col2 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col3 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col4 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col5 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col6 {
            background-color:  #058205;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col7 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col8 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col9 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col10 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col11 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col12 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col13 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col14 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col15 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col16 {
            background-color:  #088408;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col17 {
            background-color:  #088408;
        }    #T_7c84e249_9574_11e8_970b_525400123456row12_col18 {
            background-color:  #088408;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col0 {
            background-color:  #d5f6d5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col1 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col2 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col3 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col4 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col5 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col6 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col7 {
            background-color:  #048204;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col8 {
            background-color:  #048204;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col9 {
            background-color:  #048204;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col10 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col11 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col12 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col13 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col14 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col15 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col16 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col17 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row13_col18 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col0 {
            background-color:  #d5f6d5;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col1 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col2 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col3 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col4 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col5 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col6 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col7 {
            background-color:  #048204;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col8 {
            background-color:  #048204;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col9 {
            background-color:  #048204;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col10 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col11 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col12 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col13 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col14 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col15 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col16 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col17 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row14_col18 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col0 {
            background-color:  #d6f7d6;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col1 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col2 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col3 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col4 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col5 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col6 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col7 {
            background-color:  #048204;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col8 {
            background-color:  #048204;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col9 {
            background-color:  #048204;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col10 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col11 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col12 {
            background-color:  #058305;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col13 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col14 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col15 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col16 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col17 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row15_col18 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col0 {
            background-color:  #d2f4d2;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col1 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col2 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col3 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col4 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col5 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col6 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col7 {
            background-color:  #068306;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col8 {
            background-color:  #068306;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col9 {
            background-color:  #068306;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col10 {
            background-color:  #078407;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col11 {
            background-color:  #078407;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col12 {
            background-color:  #078407;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col13 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col14 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col15 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col16 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col17 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row16_col18 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col0 {
            background-color:  #d2f4d2;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col1 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col2 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col3 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col4 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col5 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col6 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col7 {
            background-color:  #068306;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col8 {
            background-color:  #068306;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col9 {
            background-color:  #068306;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col10 {
            background-color:  #078407;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col11 {
            background-color:  #078407;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col12 {
            background-color:  #078407;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col13 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col14 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col15 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col16 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col17 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row17_col18 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col0 {
            background-color:  #d3f5d3;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col1 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col2 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col3 {
            background-color:  #038103;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col4 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col5 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col6 {
            background-color:  #018001;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col7 {
            background-color:  #068306;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col8 {
            background-color:  #068306;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col9 {
            background-color:  #068306;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col10 {
            background-color:  #078407;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col11 {
            background-color:  #078407;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col12 {
            background-color:  #078407;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col13 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col14 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col15 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col16 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col17 {
            background-color:  #008000;
        }    #T_7c84e249_9574_11e8_970b_525400123456row18_col18 {
            background-color:  #008000;
        }</style>  
<table id="T_7c84e249_9574_11e8_970b_525400123456" ><caption>Correlations between size measures</caption> 
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
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row0" class="row_heading level0 row0" >vmlinux</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col0" class="data row0 col0" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col1" class="data row0 col1" >0.525318</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col2" class="data row0 col2" >0.525345</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col3" class="data row0 col3" >0.52439</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col4" class="data row0 col4" >0.538594</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col5" class="data row0 col5" >0.538442</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col6" class="data row0 col6" >0.537736</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col7" class="data row0 col7" >0.509452</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col8" class="data row0 col8" >0.50949</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col9" class="data row0 col9" >0.508168</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col10" class="data row0 col10" >0.512384</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col11" class="data row0 col11" >0.512425</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col12" class="data row0 col12" >0.51101</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col13" class="data row0 col13" >0.543367</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col14" class="data row0 col14" >0.543389</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col15" class="data row0 col15" >0.542644</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col16" class="data row0 col16" >0.550595</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col17" class="data row0 col17" >0.550614</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row0_col18" class="data row0 col18" >0.549945</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row1" class="row_heading level0 row1" >GZIP-bzImage</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col0" class="data row1 col0" >0.525318</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col1" class="data row1 col1" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col2" class="data row1 col2" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col3" class="data row1 col3" >0.99999</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col4" class="data row1 col4" >0.997491</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col5" class="data row1 col5" >0.996936</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col6" class="data row1 col6" >0.997652</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col7" class="data row1 col7" >0.997609</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col8" class="data row1 col8" >0.997611</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col9" class="data row1 col9" >0.997529</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col10" class="data row1 col10" >0.993304</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col11" class="data row1 col11" >0.993305</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col12" class="data row1 col12" >0.993217</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col13" class="data row1 col13" >0.997206</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col14" class="data row1 col14" >0.997202</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col15" class="data row1 col15" >0.997327</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col16" class="data row1 col16" >0.994154</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col17" class="data row1 col17" >0.994149</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row1_col18" class="data row1 col18" >0.994253</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row2" class="row_heading level0 row2" >GZIP-vmlinux</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col0" class="data row2 col0" >0.525345</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col1" class="data row2 col1" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col2" class="data row2 col2" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col3" class="data row2 col3" >0.999989</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col4" class="data row2 col4" >0.997496</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col5" class="data row2 col5" >0.996941</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col6" class="data row2 col6" >0.997657</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col7" class="data row2 col7" >0.997608</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col8" class="data row2 col8" >0.997609</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col9" class="data row2 col9" >0.997527</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col10" class="data row2 col10" >0.993303</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col11" class="data row2 col11" >0.993305</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col12" class="data row2 col12" >0.993215</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col13" class="data row2 col13" >0.99721</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col14" class="data row2 col14" >0.997207</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col15" class="data row2 col15" >0.997331</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col16" class="data row2 col16" >0.99416</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col17" class="data row2 col17" >0.994155</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row2_col18" class="data row2 col18" >0.994259</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row3" class="row_heading level0 row3" >GZIP</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col0" class="data row3 col0" >0.52439</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col1" class="data row3 col1" >0.99999</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col2" class="data row3 col2" >0.999989</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col3" class="data row3 col3" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col4" class="data row3 col4" >0.997311</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col5" class="data row3 col5" >0.996755</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col6" class="data row3 col6" >0.997493</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col7" class="data row3 col7" >0.997642</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col8" class="data row3 col8" >0.997643</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col9" class="data row3 col9" >0.99759</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col10" class="data row3 col10" >0.99333</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col11" class="data row3 col11" >0.993331</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col12" class="data row3 col12" >0.993273</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col13" class="data row3 col13" >0.997044</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col14" class="data row3 col14" >0.997039</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col15" class="data row3 col15" >0.997182</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col16" class="data row3 col16" >0.993925</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col17" class="data row3 col17" >0.99392</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row3_col18" class="data row3 col18" >0.99404</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row4" class="row_heading level0 row4" >BZIP2-bzImage</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col0" class="data row4 col0" >0.538594</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col1" class="data row4 col1" >0.997491</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col2" class="data row4 col2" >0.997496</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col3" class="data row4 col3" >0.997311</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col4" class="data row4 col4" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col5" class="data row4 col5" >0.999446</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col6" class="data row4 col6" >0.999989</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col7" class="data row4 col7" >0.994214</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col8" class="data row4 col8" >0.994222</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col9" class="data row4 col9" >0.993907</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col10" class="data row4 col10" >0.990641</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col11" class="data row4 col11" >0.990649</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col12" class="data row4 col12" >0.99031</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col13" class="data row4 col13" >0.998421</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col14" class="data row4 col14" >0.998422</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col15" class="data row4 col15" >0.998397</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col16" class="data row4 col16" >0.997187</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col17" class="data row4 col17" >0.997186</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row4_col18" class="data row4 col18" >0.997155</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row5" class="row_heading level0 row5" >BZIP2-vmlinux</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col0" class="data row5 col0" >0.538442</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col1" class="data row5 col1" >0.996936</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col2" class="data row5 col2" >0.996941</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col3" class="data row5 col3" >0.996755</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col4" class="data row5 col4" >0.999446</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col5" class="data row5 col5" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col6" class="data row5 col6" >0.999435</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col7" class="data row5 col7" >0.993656</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col8" class="data row5 col8" >0.993665</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col9" class="data row5 col9" >0.993349</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col10" class="data row5 col10" >0.990052</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col11" class="data row5 col11" >0.99006</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col12" class="data row5 col12" >0.98972</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col13" class="data row5 col13" >0.997878</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col14" class="data row5 col14" >0.997879</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col15" class="data row5 col15" >0.997853</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col16" class="data row5 col16" >0.996649</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col17" class="data row5 col17" >0.996648</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row5_col18" class="data row5 col18" >0.996617</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row6" class="row_heading level0 row6" >BZIP2</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col0" class="data row6 col0" >0.537736</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col1" class="data row6 col1" >0.997652</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col2" class="data row6 col2" >0.997657</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col3" class="data row6 col3" >0.997493</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col4" class="data row6 col4" >0.999989</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col5" class="data row6 col5" >0.999435</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col6" class="data row6 col6" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col7" class="data row6 col7" >0.994418</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col8" class="data row6 col8" >0.994426</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col9" class="data row6 col9" >0.99414</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col10" class="data row6 col10" >0.990839</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col11" class="data row6 col11" >0.990847</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col12" class="data row6 col12" >0.990539</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col13" class="data row6 col13" >0.998428</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col14" class="data row6 col14" >0.998427</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col15" class="data row6 col15" >0.998421</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col16" class="data row6 col16" >0.997126</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col17" class="data row6 col17" >0.997124</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row6_col18" class="data row6 col18" >0.997111</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row7" class="row_heading level0 row7" >LZMA-bzImage</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col0" class="data row7 col0" >0.509452</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col1" class="data row7 col1" >0.997609</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col2" class="data row7 col2" >0.997608</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col3" class="data row7 col3" >0.997642</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col4" class="data row7 col4" >0.994214</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col5" class="data row7 col5" >0.993656</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col6" class="data row7 col6" >0.994418</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col7" class="data row7 col7" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col8" class="data row7 col8" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col9" class="data row7 col9" >0.999982</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col10" class="data row7 col10" >0.993553</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col11" class="data row7 col11" >0.993553</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col12" class="data row7 col12" >0.993531</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col13" class="data row7 col13" >0.990992</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col14" class="data row7 col14" >0.990987</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col15" class="data row7 col15" >0.991147</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col16" class="data row7 col16" >0.986189</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col17" class="data row7 col17" >0.986184</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row7_col18" class="data row7 col18" >0.98632</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row8" class="row_heading level0 row8" >LZMA-vmlinux</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col0" class="data row8 col0" >0.50949</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col1" class="data row8 col1" >0.997611</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col2" class="data row8 col2" >0.997609</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col3" class="data row8 col3" >0.997643</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col4" class="data row8 col4" >0.994222</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col5" class="data row8 col5" >0.993665</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col6" class="data row8 col6" >0.994426</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col7" class="data row8 col7" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col8" class="data row8 col8" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col9" class="data row8 col9" >0.99998</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col10" class="data row8 col10" >0.993554</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col11" class="data row8 col11" >0.993554</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col12" class="data row8 col12" >0.99353</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col13" class="data row8 col13" >0.991</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col14" class="data row8 col14" >0.990995</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col15" class="data row8 col15" >0.991154</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col16" class="data row8 col16" >0.9862</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col17" class="data row8 col17" >0.986194</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row8_col18" class="data row8 col18" >0.986329</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row9" class="row_heading level0 row9" >LZMA</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col0" class="data row9 col0" >0.508168</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col1" class="data row9 col1" >0.997529</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col2" class="data row9 col2" >0.997527</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col3" class="data row9 col3" >0.99759</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col4" class="data row9 col4" >0.993907</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col5" class="data row9 col5" >0.993349</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col6" class="data row9 col6" >0.99414</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col7" class="data row9 col7" >0.999982</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col8" class="data row9 col8" >0.99998</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col9" class="data row9 col9" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col10" class="data row9 col10" >0.993526</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col11" class="data row9 col11" >0.993524</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col12" class="data row9 col12" >0.993543</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col13" class="data row9 col13" >0.990707</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col14" class="data row9 col14" >0.990701</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col15" class="data row9 col15" >0.990886</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col16" class="data row9 col16" >0.985815</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col17" class="data row9 col17" >0.985808</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row9_col18" class="data row9 col18" >0.985966</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row10" class="row_heading level0 row10" >XZ-bzImage</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col0" class="data row10 col0" >0.512384</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col1" class="data row10 col1" >0.993304</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col2" class="data row10 col2" >0.993303</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col3" class="data row10 col3" >0.99333</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col4" class="data row10 col4" >0.990641</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col5" class="data row10 col5" >0.990052</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col6" class="data row10 col6" >0.990839</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col7" class="data row10 col7" >0.993553</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col8" class="data row10 col8" >0.993554</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col9" class="data row10 col9" >0.993526</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col10" class="data row10 col10" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col11" class="data row10 col11" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col12" class="data row10 col12" >0.999979</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col13" class="data row10 col13" >0.987944</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col14" class="data row10 col14" >0.98794</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col15" class="data row10 col15" >0.988094</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col16" class="data row10 col16" >0.983703</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col17" class="data row10 col17" >0.983698</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row10_col18" class="data row10 col18" >0.983827</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row11" class="row_heading level0 row11" >XZ-vmlinux</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col0" class="data row11 col0" >0.512425</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col1" class="data row11 col1" >0.993305</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col2" class="data row11 col2" >0.993305</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col3" class="data row11 col3" >0.993331</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col4" class="data row11 col4" >0.990649</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col5" class="data row11 col5" >0.99006</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col6" class="data row11 col6" >0.990847</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col7" class="data row11 col7" >0.993553</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col8" class="data row11 col8" >0.993554</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col9" class="data row11 col9" >0.993524</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col10" class="data row11 col10" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col11" class="data row11 col11" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col12" class="data row11 col12" >0.999977</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col13" class="data row11 col13" >0.987952</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col14" class="data row11 col14" >0.987948</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col15" class="data row11 col15" >0.988102</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col16" class="data row11 col16" >0.983714</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col17" class="data row11 col17" >0.983709</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row11_col18" class="data row11 col18" >0.983837</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row12" class="row_heading level0 row12" >XZ</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col0" class="data row12 col0" >0.51101</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col1" class="data row12 col1" >0.993217</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col2" class="data row12 col2" >0.993215</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col3" class="data row12 col3" >0.993273</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col4" class="data row12 col4" >0.99031</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col5" class="data row12 col5" >0.98972</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col6" class="data row12 col6" >0.990539</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col7" class="data row12 col7" >0.993531</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col8" class="data row12 col8" >0.99353</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col9" class="data row12 col9" >0.993543</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col10" class="data row12 col10" >0.999979</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col11" class="data row12 col11" >0.999977</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col12" class="data row12 col12" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col13" class="data row12 col13" >0.987637</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col14" class="data row12 col14" >0.987632</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col15" class="data row12 col15" >0.987813</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col16" class="data row12 col16" >0.9833</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col17" class="data row12 col17" >0.983294</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row12_col18" class="data row12 col18" >0.983447</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row13" class="row_heading level0 row13" >LZO-bzImage</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col0" class="data row13 col0" >0.543367</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col1" class="data row13 col1" >0.997206</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col2" class="data row13 col2" >0.99721</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col3" class="data row13 col3" >0.997044</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col4" class="data row13 col4" >0.998421</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col5" class="data row13 col5" >0.997878</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col6" class="data row13 col6" >0.998428</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col7" class="data row13 col7" >0.990992</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col8" class="data row13 col8" >0.991</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col9" class="data row13 col9" >0.990707</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col10" class="data row13 col10" >0.987944</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col11" class="data row13 col11" >0.987952</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col12" class="data row13 col12" >0.987637</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col13" class="data row13 col13" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col14" class="data row13 col14" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col15" class="data row13 col15" >0.999992</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col16" class="data row13 col16" >0.999391</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col17" class="data row13 col17" >0.999389</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row13_col18" class="data row13 col18" >0.999375</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row14" class="row_heading level0 row14" >LZO-vmlinux</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col0" class="data row14 col0" >0.543389</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col1" class="data row14 col1" >0.997202</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col2" class="data row14 col2" >0.997207</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col3" class="data row14 col3" >0.997039</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col4" class="data row14 col4" >0.998422</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col5" class="data row14 col5" >0.997879</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col6" class="data row14 col6" >0.998427</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col7" class="data row14 col7" >0.990987</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col8" class="data row14 col8" >0.990995</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col9" class="data row14 col9" >0.990701</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col10" class="data row14 col10" >0.98794</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col11" class="data row14 col11" >0.987948</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col12" class="data row14 col12" >0.987632</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col13" class="data row14 col13" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col14" class="data row14 col14" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col15" class="data row14 col15" >0.999992</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col16" class="data row14 col16" >0.999392</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col17" class="data row14 col17" >0.999391</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row14_col18" class="data row14 col18" >0.999376</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row15" class="row_heading level0 row15" >LZO</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col0" class="data row15 col0" >0.542644</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col1" class="data row15 col1" >0.997327</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col2" class="data row15 col2" >0.997331</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col3" class="data row15 col3" >0.997182</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col4" class="data row15 col4" >0.998397</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col5" class="data row15 col5" >0.997853</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col6" class="data row15 col6" >0.998421</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col7" class="data row15 col7" >0.991147</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col8" class="data row15 col8" >0.991154</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col9" class="data row15 col9" >0.990886</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col10" class="data row15 col10" >0.988094</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col11" class="data row15 col11" >0.988102</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col12" class="data row15 col12" >0.987813</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col13" class="data row15 col13" >0.999992</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col14" class="data row15 col14" >0.999992</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col15" class="data row15 col15" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col16" class="data row15 col16" >0.999327</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col17" class="data row15 col17" >0.999325</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row15_col18" class="data row15 col18" >0.999325</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row16" class="row_heading level0 row16" >LZ4-bzImage</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col0" class="data row16 col0" >0.550595</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col1" class="data row16 col1" >0.994154</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col2" class="data row16 col2" >0.99416</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col3" class="data row16 col3" >0.993925</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col4" class="data row16 col4" >0.997187</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col5" class="data row16 col5" >0.996649</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col6" class="data row16 col6" >0.997126</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col7" class="data row16 col7" >0.986189</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col8" class="data row16 col8" >0.9862</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col9" class="data row16 col9" >0.985815</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col10" class="data row16 col10" >0.983703</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col11" class="data row16 col11" >0.983714</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col12" class="data row16 col12" >0.9833</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col13" class="data row16 col13" >0.999391</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col14" class="data row16 col14" >0.999392</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col15" class="data row16 col15" >0.999327</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col16" class="data row16 col16" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col17" class="data row16 col17" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row16_col18" class="data row16 col18" >0.999942</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row17" class="row_heading level0 row17" >LZ4-vmlinux</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col0" class="data row17 col0" >0.550614</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col1" class="data row17 col1" >0.994149</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col2" class="data row17 col2" >0.994155</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col3" class="data row17 col3" >0.99392</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col4" class="data row17 col4" >0.997186</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col5" class="data row17 col5" >0.996648</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col6" class="data row17 col6" >0.997124</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col7" class="data row17 col7" >0.986184</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col8" class="data row17 col8" >0.986194</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col9" class="data row17 col9" >0.985808</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col10" class="data row17 col10" >0.983698</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col11" class="data row17 col11" >0.983709</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col12" class="data row17 col12" >0.983294</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col13" class="data row17 col13" >0.999389</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col14" class="data row17 col14" >0.999391</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col15" class="data row17 col15" >0.999325</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col16" class="data row17 col16" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col17" class="data row17 col17" >1</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row17_col18" class="data row17 col18" >0.999941</td> 
    </tr>    <tr> 
        <th id="T_7c84e249_9574_11e8_970b_525400123456level0_row18" class="row_heading level0 row18" >LZ4</th> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col0" class="data row18 col0" >0.549945</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col1" class="data row18 col1" >0.994253</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col2" class="data row18 col2" >0.994259</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col3" class="data row18 col3" >0.99404</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col4" class="data row18 col4" >0.997155</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col5" class="data row18 col5" >0.996617</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col6" class="data row18 col6" >0.997111</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col7" class="data row18 col7" >0.98632</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col8" class="data row18 col8" >0.986329</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col9" class="data row18 col9" >0.985966</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col10" class="data row18 col10" >0.983827</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col11" class="data row18 col11" >0.983837</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col12" class="data row18 col12" >0.983447</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col13" class="data row18 col13" >0.999375</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col14" class="data row18 col14" >0.999376</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col15" class="data row18 col15" >0.999325</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col16" class="data row18 col16" >0.999942</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col17" class="data row18 col17" >0.999941</td> 
        <td id="T_7c84e249_9574_11e8_970b_525400123456row18_col18" class="data row18 col18" >1</td> 
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
    Int64Index: 29098 entries, 0 to 30636
    Columns: 9722 entries, X86_LOCAL_APIC to LZ4
    dtypes: int64(9722)
    memory usage: 2.1 GB





    ((29098, 9722), None)




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
 
    TESTING_SIZE=0.9 # 0.9 means 10% for training, 90% for testing
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
    1. feature DEBUG_INFO 4388 (0.159855)
    2. feature DEBUG_INFO_SPLIT 5574 (0.059258)
    3. feature UBSAN_SANITIZE_ALL 2155 (0.052640)
    4. feature GCOV_PROFILE_ALL 9666 (0.044641)
    5. feature DEBUG_INFO_REDUCED 5554 (0.034570)
    6. feature RANDOMIZE_BASE 4532 (0.027877)
    7. feature X86_NEED_RELOCS 4537 (0.021807)
    8. feature KASAN_OUTLINE 6112 (0.020916)
    9. feature UBSAN_ALIGNMENT 2161 (0.015664)
    10. feature USB_SERIAL_OPTICON 4981 (0.015174)
    11. feature KASAN 519 (0.014943)
    12. feature FW_LOADER_USER_HELPER 1164 (0.013109)
    13. feature KCOV_INSTRUMENT_ALL 6670 (0.012335)
    14. feature XFS_DEBUG 4600 (0.011956)
    15. feature MAXSMP 8647 (0.010408)
    16. feature X86_VSMP 4775 (0.009994)
    17. feature USB_GSPCA_STK1135 7535 (0.009752)
    18. feature B43_SDIO 1264 (0.008160)
    19. feature DRM_AMDGPU 2972 (0.006820)
    20. feature GDB_SCRIPTS 5610 (0.006801)
    21. feature USB_GSPCA_KONICA 6873 (0.006759)
    22. feature MODULES 5459 (0.006663)
    23. feature DEBUG_INFO_DWARF4 5590 (0.006456)
    24. feature LANMEDIA 8100 (0.006130)
    25. feature PATA_ACPI 656 (0.005883)
    26. feature DM_VERITY 2136 (0.005556)
    27. feature KALLSYMS_ABSOLUTE_PERCPU 6392 (0.005542)
    28. feature BT_WILINK 6887 (0.005537)
    29. feature DRM_DEBUG_MM_SELFTEST 2138 (0.005449)
    30. feature XFS_FS 4276 (0.005418)
    31. feature USB_GSPCA_OV519 6889 (0.005158)
    32. feature GOLDFISH_BUS 9299 (0.005045)
    33. feature SND_SOC_SN95031 8681 (0.004146)
    34. feature CW1200_WLAN_SPI 7003 (0.003945)
    35. feature BATMAN_ADV_DAT 9279 (0.003925)
    36. feature PRINTK_NMI 6416 (0.003812)
    37. feature FTRACE 9236 (0.003732)
    38. feature KEYBOARD_MPR121 5139 (0.003722)
    39. feature FB_CARMINE 5724 (0.003673)
    40. feature PHY_QCOM_USB_HS 8678 (0.003653)
    41. feature MTD_SPINAND_MT29F 3054 (0.003557)
    42. feature CRYPTO_SHA512 481 (0.003530)
    43. feature BLK_MQ_VIRTIO 2553 (0.003529)
    44. feature DRM_RADEON 2949 (0.003515)
    45. feature KEYBOARD_STMPE 5614 (0.003482)
    46. feature CFG80211 9272 (0.003426)
    47. feature CRC_CCITT 4187 (0.003386)
    48. feature CPUMASK_OFFSTACK 8650 (0.003373)
    49. feature BLK_MQ_PCI 2547 (0.003210)
    50. feature M62332 7707 (0.003209)
    51. feature SND_HDA_CORE 9470 (0.003198)
    52. feature I2C_HELPER_AUTO 5300 (0.003195)
    53. feature USB_GSPCA_CPIA1 6400 (0.003183)
    54. feature VIRTIO_CONSOLE 6311 (0.003128)
    55. feature PCMCIA_PCNET 8891 (0.003104)
    56. feature PPPOATM 6618 (0.003093)
    57. feature USB_DWC2_VERBOSE 2561 (0.003058)
    58. feature CAN_C_CAN_PCI 2877 (0.003054)
    59. feature SND_SOC_WM_HUBS 8178 (0.003001)
    60. feature FB_BACKLIGHT 3520 (0.002986)
    61. feature BRANCH_PROFILE_NONE 9287 (0.002975)
    62. feature ION_SYSTEM_HEAP 3030 (0.002882)
    63. feature EVM_LOAD_X509 5860 (0.002875)
    64. feature TCG_TIS_SPI 3531 (0.002840)
    65. feature WIRELESS_WDS 8764 (0.002835)
    66. feature DEBUG_PERF_USE_VMALLOC 7200 (0.002815)
    67. feature NETDEV_NOTIFIER_ERROR_INJECT 5555 (0.002779)
    68. feature STRICT_MODULE_RWX 9635 (0.002732)
    69. feature N_GSM 7507 (0.002700)
    70. feature ZD1211RW 7436 (0.002681)
    71. feature VIDEO_VIA_CAMERA 9189 (0.002680)
    72. feature INTEL_BXT_PMIC_THERMAL 7955 (0.002662)
    73. feature USB_GSPCA_SQ905C 7526 (0.002643)
    74. feature SND_SOC_RT5640 8726 (0.002627)
    75. feature MXM_WMI 3018 (0.002607)
    76. feature MMU_NOTIFIER 1947 (0.002605)
    77. feature OCFS2_FS 4623 (0.002542)
    78. feature RANDOMIZE_MEMORY 4561 (0.002516)
    79. feature ACPI_BUTTON 6258 (0.002510)
    80. feature SENSORS_ADT7X10 847 (0.002469)
    81. feature GPIO_PCA953X 4983 (0.002448)
    82. feature PM_SLEEP_DEBUG 5820 (0.002446)
    83. feature LOCK_STAT 7742 (0.002433)
    84. feature PAGE_EXTENSION 1940 (0.002427)
    85. feature CRYPTO_DEV_CHELSIO 5073 (0.002410)
    86. feature CHARGER_88PM860X 9243 (0.002379)
    87. feature NET_VENDOR_HP 4648 (0.002354)
    88. feature LTC2497 5413 (0.002321)
    89. feature MT7601U 5144 (0.002317)
    90. feature AF_RXRPC 6905 (0.002257)
    91. feature USB_EZUSB_FX2 3142 (0.002175)
    92. feature SCSI_SPI_ATTRS 249 (0.002174)
    93. feature SCSI_FC_ATTRS 253 (0.002162)
    94. feature BT_HCIBTSDIO 5990 (0.002146)
    95. feature SERIAL_ALTERA_UART 8999 (0.002125)
    96. feature LEDS_TRIGGER_TIMER 6502 (0.002045)
    97. feature BATTERY_WM97XX 8593 (0.002031)
    98. feature APPLICOM 2615 (0.002001)
    99. feature PPPOL2TP 6627 (0.001980)
    100. feature INTEL_MENLOW 7949 (0.001972)
    Prediction score (MAE): 11.02
    Prediction score (MSE): 1291492941.96
    Prediction score (R2): 0.71
    Prediction score (MRE): 18.33



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
    Int64Index: 29098 entries, 0 to 30636
    Columns: 9722 entries, X86_LOCAL_APIC to LZ4
    dtypes: int64(9722)
    memory usage: 2.1 GB



```python
rawtuxdata['vmlinux'].sort_values()
```




    12299       7317008
    20106      10572520
    13561      10856456
    19509      10857304
    25873      10865408
    28635      10889536
    29874      10892608
    20796      10918504
    17312      10955056
    30104      10962488
    21457      10981392
    17894      10992912
    6849       11035016
    26114      11035128
    3182       11098984
    6067       11100872
    27090      11105984
    28615      11116560
    20800      11117600
    16624      11120888
    5583       11123472
    23444      11129920
    26322      11148984
    1972       11150720
    23329      11161272
    12617      11168320
    17864      11170448
    24818      11191840
    29917      11193720
    19632      11200416
                ...    
    8253      832672688
    901       841872848
    5243      843284784
    8813      855732544
    3960      874627784
    10917     876365640
    7032      888694960
    1039      895307512
    972       897223336
    11807     906055848
    4951      913548408
    6788      918916232
    2256      921336008
    9121      935902976
    2177      937690792
    7218      939015144
    5928      946132424
    3929      971560760
    7449      972595136
    11377    1008068400
    5031     1037628968
    8302     1072199216
    7314     1093119496
    8054     1172001672
    6932     1212368184
    1950     1214925624
    7760     1321293272
    11831    1368188424
    10425    1407581792
    10902    1693673912
    Name: vmlinux, Length: 29098, dtype: int64


