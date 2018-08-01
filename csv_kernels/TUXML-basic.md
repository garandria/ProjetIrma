

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

    /usr/lib/python3/dist-packages/IPython/core/interactiveshell.py:2718: DtypeWarning: Columns (1150,6015,6026,7676,7726) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)


    Original size (#configs/#options) of the dataset (12500, 12798)
    Number of options with only one value (eg always y): (3326, 1)
    Non tri-state value options (eg string or integer or hybrid values): (155, 1) 
    Predictor variables: 9317



```python
'X86_64' in ftuniques, 'DEBUG_INFO' in ftuniques, 'GCOV_PROFILE_ALL' in ftuniques, 'KASAN' in ftuniques, 'UBSAN_SANITIZE_ALL' in ftuniques, 'RELOCATABLE' in ftuniques, 'XFS_DEBUG' in ftuniques, 'AIC7XXX_BUILD_FIRMWARE' in ftuniques, 'AIC79XX_BUILD_FIRMWARE' in ftuniques, 'WANXL_BUILD_FIRMWARE' in ftuniques
```




    (False, True, False, True, False, False, False, False, False, False)




```python
if 'RELOCATABLE' in rawtuxdata.columns:
    print(rawtuxdata.query("RELOCATABLE == 'y'")[['cid', 'RELOCATABLE']])
```

             cid RELOCATABLE
    0      80000           y
    4      80004           y
    5      80005           y
    6      80006           y
    8      80008           y
    10     80010           y
    11     80011           y
    12     80012           y
    14     80014           y
    21     80021           y
    23     80023           y
    24     80024           y
    27     80027           y
    30     80030           y
    31     80031           y
    32     80032           y
    33     80033           y
    36     80036           y
    39     80039           y
    41     80041           y
    43     80043           y
    45     80045           y
    46     80046           y
    49     80049           y
    50     80050           y
    51     80051           y
    52     80052           y
    55     80055           y
    56     80056           y
    58     80058           y
    ...      ...         ...
    12229  92229           y
    12237  92237           y
    12241  92241           y
    12251  92251           y
    12253  92253           y
    12255  92255           y
    12276  92276           y
    12277  92277           y
    12283  92283           y
    12290  92290           y
    12291  92291           y
    12293  92293           y
    12303  92303           y
    12353  92353           y
    12357  92357           y
    12398  92398           y
    12402  92402           y
    12405  92405           y
    12419  92419           y
    12421  92421           y
    12434  92434           y
    12439  92439           y
    12445  92445           y
    12455  92455           y
    12461  92461           y
    12465  92465           y
    12468  92468           y
    12473  92473           y
    12478  92478           y
    12491  92491           y
    
    [1520 rows x 2 columns]



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
      <th>13</th>
      <td>80013</td>
      <td>2018-07-25 11:19:18</td>
      <td>700.759</td>
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
      <th>32</th>
      <td>80032</td>
      <td>2018-07-25 11:22:57</td>
      <td>688.759</td>
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
      <th>36</th>
      <td>80036</td>
      <td>2018-07-25 11:24:24</td>
      <td>336.431</td>
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
      <td>y</td>
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>54</th>
      <td>80054</td>
      <td>2018-07-25 11:28:52</td>
      <td>290.582</td>
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
      <th>58</th>
      <td>80058</td>
      <td>2018-07-25 11:29:44</td>
      <td>450.351</td>
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
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>82</th>
      <td>80082</td>
      <td>2018-07-25 11:35:20</td>
      <td>377.517</td>
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
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>95</th>
      <td>80095</td>
      <td>2018-07-25 11:42:22</td>
      <td>599.840</td>
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
      <td>m</td>
      <td>y</td>
    </tr>
    <tr>
      <th>119</th>
      <td>80119</td>
      <td>2018-07-25 11:51:50</td>
      <td>418.584</td>
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
      <th>120</th>
      <td>80120</td>
      <td>2018-07-25 11:52:05</td>
      <td>630.787</td>
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
      <td>y</td>
    </tr>
    <tr>
      <th>139</th>
      <td>80139</td>
      <td>2018-07-25 11:58:24</td>
      <td>450.553</td>
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
      <th>143</th>
      <td>80143</td>
      <td>2018-07-25 12:00:22</td>
      <td>404.861</td>
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
      <td>m</td>
      <td>n</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
    </tr>
    <tr>
      <th>156</th>
      <td>80156</td>
      <td>2018-07-25 12:05:40</td>
      <td>526.263</td>
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
      <th>171</th>
      <td>80171</td>
      <td>2018-07-25 12:10:56</td>
      <td>341.523</td>
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
      <th>204</th>
      <td>80204</td>
      <td>2018-07-25 12:22:54</td>
      <td>487.166</td>
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
      <th>206</th>
      <td>80206</td>
      <td>2018-07-25 12:23:01</td>
      <td>240.532</td>
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
      <td>m</td>
      <td>n</td>
    </tr>
    <tr>
      <th>210</th>
      <td>80210</td>
      <td>2018-07-25 12:24:38</td>
      <td>469.227</td>
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
      <th>225</th>
      <td>80225</td>
      <td>2018-07-25 12:30:31</td>
      <td>536.526</td>
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
      <th>233</th>
      <td>80233</td>
      <td>2018-07-25 12:33:32</td>
      <td>495.983</td>
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
      <th>251</th>
      <td>80251</td>
      <td>2018-07-25 12:39:24</td>
      <td>339.134</td>
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
      <th>253</th>
      <td>80253</td>
      <td>2018-07-25 12:40:12</td>
      <td>457.590</td>
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
      <th>263</th>
      <td>80263</td>
      <td>2018-07-25 12:43:26</td>
      <td>249.856</td>
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
      <th>298</th>
      <td>80298</td>
      <td>2018-07-25 12:57:14</td>
      <td>276.900</td>
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
      <td>y</td>
    </tr>
    <tr>
      <th>302</th>
      <td>80302</td>
      <td>2018-07-25 12:57:49</td>
      <td>478.828</td>
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
      <td>m</td>
      <td>n</td>
      <td>n</td>
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>303</th>
      <td>80303</td>
      <td>2018-07-25 12:58:02</td>
      <td>518.200</td>
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
      <td>y</td>
      <td>n</td>
    </tr>
    <tr>
      <th>311</th>
      <td>80311</td>
      <td>2018-07-25 12:59:33</td>
      <td>719.122</td>
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
      <td>m</td>
      <td>n</td>
      <td>m</td>
      <td>y</td>
    </tr>
    <tr>
      <th>322</th>
      <td>80322</td>
      <td>2018-07-25 13:03:20</td>
      <td>431.227</td>
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
      <td>n</td>
      <td>n</td>
    </tr>
    <tr>
      <th>361</th>
      <td>80361</td>
      <td>2018-07-25 13:17:02</td>
      <td>483.506</td>
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
      <td>m</td>
      <td>n</td>
    </tr>
    <tr>
      <th>371</th>
      <td>80371</td>
      <td>2018-07-25 13:20:10</td>
      <td>659.851</td>
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
      <th>380</th>
      <td>80380</td>
      <td>2018-07-25 13:22:25</td>
      <td>501.904</td>
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
      <td>y</td>
      <td>m</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
      <td>y</td>
    </tr>
    <tr>
      <th>406</th>
      <td>80406</td>
      <td>2018-07-25 13:31:12</td>
      <td>437.189</td>
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
      <th>6957</th>
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
      <th>7044</th>
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
      <th>7074</th>
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
      <th>7340</th>
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
      <th>7374</th>
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
      <th>7514</th>
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
      <th>7566</th>
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
      <th>7650</th>
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
      <th>8140</th>
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
      <th>8152</th>
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
      <th>8252</th>
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
      <th>8748</th>
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
      <th>8817</th>
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
      <th>8909</th>
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
      <th>9577</th>
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
      <th>9618</th>
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
      <th>9701</th>
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
      <th>9757</th>
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
      <th>9854</th>
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
      <th>10506</th>
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
      <th>10710</th>
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
      <th>10975</th>
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
      <th>11054</th>
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
      <th>11189</th>
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
      <th>11291</th>
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
      <th>11633</th>
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
      <th>11742</th>
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
      <th>11862</th>
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
      <th>12072</th>
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
      <th>12120</th>
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
<p>418 rows × 9472 columns</p>
</div>




```python
#rawtuxdata[rawtuxdata['vmlinux'] == 1168072]['MODULES']
rawtuxdata.query("vmlinux == 1168072")['MODULES'] #tiny config for X86_32
```




    6717    n
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




    0     33069672
    1     19186776
    2     22709920
    3     21871888
    4     67107680
    5     32868392
    6     16425352
    7     50971136
    8     30877584
    9     27698128
    10    30303952
    11    18044064
    12    25690984
    13          -1
    14    33597264
    15    32946088
    16    45450008
    17    29319792
    18    21469760
    19    24365904
    Name: vmlinux, dtype: int64




```python
rawtuxdata.shape, rawtuxdata.query("vmlinux != -1").shape
```




    ((12500, 9472), (12082, 9472))




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
    count     12500
    unique        2
    top           y
    freq      12499
    Name: X86_64, dtype: object
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 12499 entries, 0 to 12499
    Columns: 9472 entries, cid to NETWORK_FILESYSTEMS
    dtypes: float64(1), int64(140), object(9331)
    memory usage: 7.2 GB



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

    12499 before the removal of some entries (those with same configurations)
    12496 after the removal of some entries (those with same configurations)
    12079 after the removal of configurations that do NOT compile



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
<p>0 rows × 9472 columns</p>
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
      <th>8035</th>
      <td>88035</td>
      <td>2018-07-30 10:24:57</td>
      <td>29.6678</td>
      <td>7317008</td>
      <td>646608</td>
      <td>2733176</td>
      <td>501235</td>
      <td>4718032</td>
      <td>6804048</td>
      <td>458475</td>
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
<p>1 rows × 9472 columns</p>
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





    count    1.207900e+04
    mean     3.052792e+07
    std      1.324927e+07
    min      7.317008e+06
    25%      2.170131e+07
    50%      2.741326e+07
    75%      3.624314e+07
    max      2.509113e+08
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
<p>0 rows × 9472 columns</p>
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
    #T_4a476e6e_9562_11e8_a1ed_525400123456row0_col0 {
            color:  black;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row0_col1 {
            color:  green;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row0_col2 {
            color:  green;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row0_col3 {
            color:  green;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row0_col4 {
            color:  red;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row0_col5 {
            color:  red;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row1_col0 {
            color:  red;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row1_col1 {
            color:  black;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row1_col2 {
            color:  green;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row1_col3 {
            color:  green;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row1_col4 {
            color:  red;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row1_col5 {
            color:  red;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row2_col0 {
            color:  red;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row2_col1 {
            color:  red;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row2_col2 {
            color:  black;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row2_col3 {
            color:  green;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row2_col4 {
            color:  red;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row2_col5 {
            color:  red;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row3_col0 {
            color:  red;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row3_col1 {
            color:  red;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row3_col2 {
            color:  red;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row3_col3 {
            color:  black;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row3_col4 {
            color:  red;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row3_col5 {
            color:  red;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row4_col0 {
            color:  green;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row4_col1 {
            color:  green;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row4_col2 {
            color:  green;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row4_col3 {
            color:  green;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row4_col4 {
            color:  black;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row4_col5 {
            color:  red;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row5_col0 {
            color:  green;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row5_col1 {
            color:  green;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row5_col2 {
            color:  green;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row5_col3 {
            color:  green;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row5_col4 {
            color:  green;
        }    #T_4a476e6e_9562_11e8_a1ed_525400123456row5_col5 {
            color:  black;
        }</style>  
<table id="T_4a476e6e_9562_11e8_a1ed_525400123456" ><caption>Difference (average in percentage) per compression methods</caption> 
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
        <th id="T_4a476e6e_9562_11e8_a1ed_525400123456level0_row0" class="row_heading level0 row0" >GZIPo</th> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row0_col0" class="data row0 col0" >0</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row0_col1" class="data row0 col1" >3.14026</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row0_col2" class="data row0 col2" >22.6425</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row0_col3" class="data row0 col3" >35.2092</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row0_col4" class="data row0 col4" >-9.16576</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row0_col5" class="data row0 col5" >-90967.7</td> 
    </tr>    <tr> 
        <th id="T_4a476e6e_9562_11e8_a1ed_525400123456level0_row1" class="row_heading level0 row1" >BZIP2o</th> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row1_col0" class="data row1 col0" >-3.02857</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row1_col1" class="data row1 col1" >0</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row1_col2" class="data row1 col2" >18.9298</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row1_col3" class="data row1 col3" >31.1257</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row1_col4" class="data row1 col4" >-11.9251</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row1_col5" class="data row1 col5" >-87466.9</td> 
    </tr>    <tr> 
        <th id="T_4a476e6e_9562_11e8_a1ed_525400123456level0_row2" class="row_heading level0 row2" >LZMAo</th> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row2_col0" class="data row2 col0" >-18.4501</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row2_col1" class="data row2 col1" >-15.8883</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row2_col2" class="data row2 col2" >0</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row2_col3" class="data row2 col3" >10.2555</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row2_col4" class="data row2 col4" >-25.92</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row2_col5" class="data row2 col5" >-71482.4</td> 
    </tr>    <tr> 
        <th id="T_4a476e6e_9562_11e8_a1ed_525400123456level0_row3" class="row_heading level0 row3" >XZo</th> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row3_col0" class="data row3 col0" >-25.783</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row3_col1" class="data row3 col1" >-23.4461</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row3_col2" class="data row3 col2" >-8.98553</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row3_col3" class="data row3 col3" >0</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row3_col4" class="data row3 col4" >-32.5879</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row3_col5" class="data row3 col5" >-66240.3</td> 
    </tr>    <tr> 
        <th id="T_4a476e6e_9562_11e8_a1ed_525400123456level0_row4" class="row_heading level0 row4" >LZOo</th> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row4_col0" class="data row4 col0" >10.1027</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row4_col1" class="data row4 col1" >13.5494</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row4_col2" class="data row4 col2" >35.0421</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row4_col3" class="data row4 col3" >48.863</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row4_col4" class="data row4 col4" >0</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row4_col5" class="data row4 col5" >-101100</td> 
    </tr>    <tr> 
        <th id="T_4a476e6e_9562_11e8_a1ed_525400123456level0_row5" class="row_heading level0 row5" >LZ4o</th> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row5_col0" class="data row5 col0" >17.321</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row5_col1" class="data row5 col1" >20.987</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row5_col2" class="data row5 col2" >43.8997</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row5_col3" class="data row5 col3" >58.6362</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row5_col4" class="data row5 col4" >6.55089</td> 
        <td id="T_4a476e6e_9562_11e8_a1ed_525400123456row5_col5" class="data row5 col5" >0</td> 
    </tr></tbody> 
</table> 




```python
compareCompress("-bzImage").style.set_caption('Difference (average in percentage) per compression methods, bzImage').applymap(color_negative_positive)

```




<style  type="text/css" >
    #T_4a476e6f_9562_11e8_a1ed_525400123456row0_col0 {
            color:  black;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row0_col1 {
            color:  red;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row0_col2 {
            color:  green;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row0_col3 {
            color:  green;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row0_col4 {
            color:  red;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row0_col5 {
            color:  red;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row1_col0 {
            color:  green;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row1_col1 {
            color:  black;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row1_col2 {
            color:  green;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row1_col3 {
            color:  green;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row1_col4 {
            color:  green;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row1_col5 {
            color:  green;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row2_col0 {
            color:  red;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row2_col1 {
            color:  red;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row2_col2 {
            color:  black;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row2_col3 {
            color:  green;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row2_col4 {
            color:  red;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row2_col5 {
            color:  red;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row3_col0 {
            color:  red;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row3_col1 {
            color:  red;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row3_col2 {
            color:  red;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row3_col3 {
            color:  black;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row3_col4 {
            color:  red;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row3_col5 {
            color:  red;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row4_col0 {
            color:  green;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row4_col1 {
            color:  red;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row4_col2 {
            color:  green;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row4_col3 {
            color:  green;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row4_col4 {
            color:  black;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row4_col5 {
            color:  red;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row5_col0 {
            color:  green;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row5_col1 {
            color:  red;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row5_col2 {
            color:  green;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row5_col3 {
            color:  green;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row5_col4 {
            color:  green;
        }    #T_4a476e6f_9562_11e8_a1ed_525400123456row5_col5 {
            color:  black;
        }</style>  
<table id="T_4a476e6f_9562_11e8_a1ed_525400123456" ><caption>Difference (average in percentage) per compression methods, bzImage</caption> 
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
        <th id="T_4a476e6f_9562_11e8_a1ed_525400123456level0_row0" class="row_heading level0 row0" >GZIPo</th> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row0_col0" class="data row0 col0" >0</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row0_col1" class="data row0 col1" >-35.3384</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row0_col2" class="data row0 col2" >22.2595</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row0_col3" class="data row0 col3" >34.2024</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row0_col4" class="data row0 col4" >-8.81806</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row0_col5" class="data row0 col5" >-14.4595</td> 
    </tr>    <tr> 
        <th id="T_4a476e6f_9562_11e8_a1ed_525400123456level0_row1" class="row_heading level0 row1" >BZIP2o</th> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row1_col0" class="data row1 col0" >58.8322</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row1_col1" class="data row1 col1" >0</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row1_col2" class="data row1 col2" >93.9605</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row1_col3" class="data row1 col3" >112.417</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row1_col4" class="data row1 col4" >44.8345</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row1_col5" class="data row1 col5" >35.8698</td> 
    </tr>    <tr> 
        <th id="T_4a476e6f_9562_11e8_a1ed_525400123456level0_row2" class="row_heading level0 row2" >LZMAo</th> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row2_col0" class="data row2 col0" >-18.1941</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row2_col1" class="data row2 col1" >-47.1627</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row2_col2" class="data row2 col2" >0</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row2_col3" class="data row2 col3" >9.77394</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row2_col4" class="data row2 col4" >-25.4027</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row2_col5" class="data row2 col5" >-30.0163</td> 
    </tr>    <tr> 
        <th id="T_4a476e6f_9562_11e8_a1ed_525400123456level0_row3" class="row_heading level0 row3" >XZo</th> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row3_col0" class="data row3 col0" >-25.2321</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row3_col1" class="data row3 col1" >-51.8049</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row3_col2" class="data row3 col2" >-8.59972</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row3_col3" class="data row3 col3" >0</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row3_col4" class="data row3 col4" >-31.8258</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row3_col5" class="data row3 col5" >-36.039</td> 
    </tr>    <tr> 
        <th id="T_4a476e6f_9562_11e8_a1ed_525400123456level0_row4" class="row_heading level0 row4" >LZOo</th> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row4_col0" class="data row4 col0" >9.68193</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row4_col1" class="data row4 col1" >-29.0747</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row4_col2" class="data row4 col2" >34.1063</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row4_col3" class="data row4 col3" >47.194</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row4_col4" class="data row4 col4" >0</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row4_col5" class="data row4 col5" >-6.19106</td> 
    </tr>    <tr> 
        <th id="T_4a476e6f_9562_11e8_a1ed_525400123456level0_row5" class="row_heading level0 row5" >LZ4o</th> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row5_col0" class="data row5 col0" >16.9292</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row5_col1" class="data row5 col1" >-24.3902</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row5_col2" class="data row5 col2" >42.9713</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row5_col3" class="data row5 col3" >56.932</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row5_col4" class="data row5 col4" >6.60268</td> 
        <td id="T_4a476e6f_9562_11e8_a1ed_525400123456row5_col5" class="data row5 col5" >0</td> 
    </tr></tbody> 
</table> 




```python
compareCompress("-vmlinux").style.set_caption('Difference (average in percentage) per compression methods, vmlinux').applymap(color_negative_positive)

```




<style  type="text/css" >
    #T_4a476e70_9562_11e8_a1ed_525400123456row0_col0 {
            color:  black;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row0_col1 {
            color:  red;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row0_col2 {
            color:  green;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row0_col3 {
            color:  green;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row0_col4 {
            color:  red;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row0_col5 {
            color:  red;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row1_col0 {
            color:  green;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row1_col1 {
            color:  black;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row1_col2 {
            color:  green;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row1_col3 {
            color:  green;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row1_col4 {
            color:  green;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row1_col5 {
            color:  green;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row2_col0 {
            color:  red;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row2_col1 {
            color:  red;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row2_col2 {
            color:  black;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row2_col3 {
            color:  green;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row2_col4 {
            color:  red;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row2_col5 {
            color:  red;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row3_col0 {
            color:  red;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row3_col1 {
            color:  red;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row3_col2 {
            color:  red;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row3_col3 {
            color:  black;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row3_col4 {
            color:  red;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row3_col5 {
            color:  red;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row4_col0 {
            color:  green;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row4_col1 {
            color:  red;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row4_col2 {
            color:  green;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row4_col3 {
            color:  green;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row4_col4 {
            color:  black;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row4_col5 {
            color:  red;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row5_col0 {
            color:  green;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row5_col1 {
            color:  red;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row5_col2 {
            color:  green;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row5_col3 {
            color:  green;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row5_col4 {
            color:  green;
        }    #T_4a476e70_9562_11e8_a1ed_525400123456row5_col5 {
            color:  black;
        }</style>  
<table id="T_4a476e70_9562_11e8_a1ed_525400123456" ><caption>Difference (average in percentage) per compression methods, vmlinux</caption> 
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
        <th id="T_4a476e70_9562_11e8_a1ed_525400123456level0_row0" class="row_heading level0 row0" >GZIPo</th> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row0_col0" class="data row0 col0" >0</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row0_col1" class="data row0 col1" >-29.3512</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row0_col2" class="data row0 col2" >16.3982</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row0_col3" class="data row0 col3" >24.6336</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row0_col4" class="data row0 col4" >-6.91926</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row0_col5" class="data row0 col5" >-11.4992</td> 
    </tr>    <tr> 
        <th id="T_4a476e70_9562_11e8_a1ed_525400123456level0_row1" class="row_heading level0 row1" >BZIP2o</th> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row1_col0" class="data row1 col0" >43.1561</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row1_col1" class="data row1 col1" >0</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row1_col2" class="data row1 col2" >66.2646</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row1_col3" class="data row1 col3" >77.7221</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row1_col4" class="data row1 col4" >33.3474</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row1_col5" class="data row1 col5" >26.8432</td> 
    </tr>    <tr> 
        <th id="T_4a476e70_9562_11e8_a1ed_525400123456level0_row2" class="row_heading level0 row2" >LZMAo</th> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row2_col0" class="data row2 col0" >-14.0449</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row2_col1" class="data row2 col1" >-39.4059</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row2_col2" class="data row2 col2" >0</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row2_col3" class="data row2 col3" >7.04671</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row2_col4" class="data row2 col4" >-19.977</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row2_col5" class="data row2 col5" >-23.9062</td> 
    </tr>    <tr> 
        <th id="T_4a476e70_9562_11e8_a1ed_525400123456level0_row3" class="row_heading level0 row3" >XZo</th> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row3_col0" class="data row3 col0" >-19.5245</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row3_col1" class="data row3 col1" >-43.3615</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row3_col2" class="data row3 col2" >-6.39961</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row3_col3" class="data row3 col3" >0</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row3_col4" class="data row3 col4" >-25.0735</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row3_col5" class="data row3 col5" >-28.7452</td> 
    </tr>    <tr> 
        <th id="T_4a476e70_9562_11e8_a1ed_525400123456level0_row4" class="row_heading level0 row4" >LZOo</th> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row4_col0" class="data row4 col0" >7.44653</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row4_col1" class="data row4 col1" >-24.0365</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row4_col2" class="data row4 col2" >25.0904</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row4_col3" class="data row4 col3" >33.9492</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row4_col4" class="data row4 col4" >0</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row4_col5" class="data row4 col5" >-4.92634</td> 
    </tr>    <tr> 
        <th id="T_4a476e70_9562_11e8_a1ed_525400123456level0_row5" class="row_heading level0 row5" >LZ4o</th> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row5_col0" class="data row5 col0" >13.0258</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row5_col1" class="data row5 col1" >-20.0569</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row5_col2" class="data row5 col2" >31.6006</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row5_col3" class="data row5 col3" >40.9349</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row5_col4" class="data row5 col4" >5.1858</td> 
        <td id="T_4a476e70_9562_11e8_a1ed_525400123456row5_col5" class="data row5 col5" >0</td> 
    </tr></tbody> 
</table> 




```python
cm = sns.light_palette("green", as_cmap=True)
pd.DataFrame.corr(rawtuxdata[size_methods]).style.set_caption('Correlations between size measures').background_gradient(cmap=cm)

```




<style  type="text/css" >
    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col0 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col1 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col2 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col3 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col4 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col5 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col6 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col7 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col8 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col9 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col10 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col11 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col12 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col13 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col14 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col15 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col16 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col17 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row0_col18 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col0 {
            background-color:  #d0f3d0;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col1 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col2 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col3 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col4 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col5 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col6 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col7 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col8 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col9 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col10 {
            background-color:  #0f880f;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col11 {
            background-color:  #0f880f;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col12 {
            background-color:  #0f880f;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col13 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col14 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col15 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col16 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col17 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row1_col18 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col0 {
            background-color:  #d0f3d0;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col1 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col2 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col3 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col4 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col5 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col6 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col7 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col8 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col9 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col10 {
            background-color:  #0f880f;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col11 {
            background-color:  #0f880f;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col12 {
            background-color:  #0f880f;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col13 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col14 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col15 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col16 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col17 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row2_col18 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col0 {
            background-color:  #d3f5d3;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col1 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col2 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col3 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col4 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col5 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col6 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col7 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col8 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col9 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col10 {
            background-color:  #0f880f;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col11 {
            background-color:  #0f880f;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col12 {
            background-color:  #0f880f;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col13 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col14 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col15 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col16 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col17 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row3_col18 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col0 {
            background-color:  #c2ecc2;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col1 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col2 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col3 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col4 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col5 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col6 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col7 {
            background-color:  #028102;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col8 {
            background-color:  #028102;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col9 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col10 {
            background-color:  #128a12;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col11 {
            background-color:  #128a12;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col12 {
            background-color:  #128a12;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col13 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col14 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col15 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col16 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col17 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row4_col18 {
            background-color:  #028102;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col0 {
            background-color:  #c4edc4;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col1 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col2 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col3 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col4 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col5 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col6 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col7 {
            background-color:  #058205;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col8 {
            background-color:  #058205;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col9 {
            background-color:  #058305;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col10 {
            background-color:  #158b15;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col11 {
            background-color:  #158b15;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col12 {
            background-color:  #158b15;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col13 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col14 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col15 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col16 {
            background-color:  #058205;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col17 {
            background-color:  #058205;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row5_col18 {
            background-color:  #058305;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col0 {
            background-color:  #c4edc4;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col1 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col2 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col3 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col4 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col5 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col6 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col7 {
            background-color:  #028102;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col8 {
            background-color:  #028102;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col9 {
            background-color:  #028102;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col10 {
            background-color:  #128a12;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col11 {
            background-color:  #128a12;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col12 {
            background-color:  #128a12;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col13 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col14 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col15 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col16 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col17 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row6_col18 {
            background-color:  #028102;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col0 {
            background-color:  #e3fee3;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col1 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col2 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col3 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col4 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col5 {
            background-color:  #058305;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col6 {
            background-color:  #028102;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col7 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col8 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col9 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col10 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col11 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col12 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col13 {
            background-color:  #058205;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col14 {
            background-color:  #058205;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col15 {
            background-color:  #058205;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col16 {
            background-color:  #078407;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col17 {
            background-color:  #078407;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row7_col18 {
            background-color:  #078407;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col0 {
            background-color:  #e3fee3;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col1 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col2 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col3 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col4 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col5 {
            background-color:  #058305;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col6 {
            background-color:  #028102;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col7 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col8 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col9 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col10 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col11 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col12 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col13 {
            background-color:  #058205;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col14 {
            background-color:  #058205;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col15 {
            background-color:  #058205;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col16 {
            background-color:  #078407;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col17 {
            background-color:  #078407;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row8_col18 {
            background-color:  #078407;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col0 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col1 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col2 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col3 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col4 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col5 {
            background-color:  #068306;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col6 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col7 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col8 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col9 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col10 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col11 {
            background-color:  #118911;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col12 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col13 {
            background-color:  #058205;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col14 {
            background-color:  #058205;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col15 {
            background-color:  #058205;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col16 {
            background-color:  #088408;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col17 {
            background-color:  #088408;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row9_col18 {
            background-color:  #088408;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col0 {
            background-color:  #e2fde2;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col1 {
            background-color:  #118911;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col2 {
            background-color:  #118911;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col3 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col4 {
            background-color:  #158b15;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col5 {
            background-color:  #178d17;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col6 {
            background-color:  #158b15;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col7 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col8 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col9 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col10 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col11 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col12 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col13 {
            background-color:  #158b15;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col14 {
            background-color:  #158b15;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col15 {
            background-color:  #148b14;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col16 {
            background-color:  #188d18;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col17 {
            background-color:  #188d18;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row10_col18 {
            background-color:  #188d18;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col0 {
            background-color:  #e2fde2;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col1 {
            background-color:  #118911;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col2 {
            background-color:  #118911;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col3 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col4 {
            background-color:  #158b15;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col5 {
            background-color:  #178d17;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col6 {
            background-color:  #158b15;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col7 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col8 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col9 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col10 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col11 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col12 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col13 {
            background-color:  #158b15;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col14 {
            background-color:  #158b15;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col15 {
            background-color:  #148b14;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col16 {
            background-color:  #188d18;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col17 {
            background-color:  #188d18;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row11_col18 {
            background-color:  #188d18;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col0 {
            background-color:  #e5ffe5;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col1 {
            background-color:  #118911;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col2 {
            background-color:  #118911;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col3 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col4 {
            background-color:  #158b15;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col5 {
            background-color:  #188d18;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col6 {
            background-color:  #158b15;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col7 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col8 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col9 {
            background-color:  #108910;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col10 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col11 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col12 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col13 {
            background-color:  #158b15;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col14 {
            background-color:  #158b15;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col15 {
            background-color:  #158b15;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col16 {
            background-color:  #198e19;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col17 {
            background-color:  #198e19;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row12_col18 {
            background-color:  #198e19;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col0 {
            background-color:  #b9e7b9;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col1 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col2 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col3 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col4 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col5 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col6 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col7 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col8 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col9 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col10 {
            background-color:  #118911;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col11 {
            background-color:  #118911;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col12 {
            background-color:  #118911;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col13 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col14 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col15 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col16 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col17 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row13_col18 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col0 {
            background-color:  #b9e7b9;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col1 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col2 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col3 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col4 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col5 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col6 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col7 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col8 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col9 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col10 {
            background-color:  #118911;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col11 {
            background-color:  #118911;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col12 {
            background-color:  #118911;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col13 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col14 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col15 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col16 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col17 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row14_col18 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col0 {
            background-color:  #bbe8bb;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col1 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col2 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col3 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col4 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col5 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col6 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col7 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col8 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col9 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col10 {
            background-color:  #118911;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col11 {
            background-color:  #118911;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col12 {
            background-color:  #118911;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col13 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col14 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col15 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col16 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col17 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row15_col18 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col0 {
            background-color:  #b1e2b1;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col1 {
            background-color:  #028102;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col2 {
            background-color:  #028102;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col3 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col4 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col5 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col6 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col7 {
            background-color:  #058305;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col8 {
            background-color:  #058305;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col9 {
            background-color:  #058305;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col10 {
            background-color:  #138a13;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col11 {
            background-color:  #138a13;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col12 {
            background-color:  #138a13;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col13 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col14 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col15 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col16 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col17 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row16_col18 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col0 {
            background-color:  #b1e2b1;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col1 {
            background-color:  #028102;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col2 {
            background-color:  #028102;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col3 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col4 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col5 {
            background-color:  #048204;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col6 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col7 {
            background-color:  #058305;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col8 {
            background-color:  #058305;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col9 {
            background-color:  #058305;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col10 {
            background-color:  #138a13;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col11 {
            background-color:  #138a13;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col12 {
            background-color:  #138a13;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col13 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col14 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col15 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col16 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col17 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row17_col18 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col0 {
            background-color:  #b4e4b4;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col1 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col2 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col3 {
            background-color:  #038103;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col4 {
            background-color:  #028102;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col5 {
            background-color:  #058205;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col6 {
            background-color:  #028102;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col7 {
            background-color:  #058305;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col8 {
            background-color:  #058305;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col9 {
            background-color:  #068306;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col10 {
            background-color:  #148b14;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col11 {
            background-color:  #148b14;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col12 {
            background-color:  #148b14;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col13 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col14 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col15 {
            background-color:  #018001;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col16 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col17 {
            background-color:  #008000;
        }    #T_4a476e71_9562_11e8_a1ed_525400123456row18_col18 {
            background-color:  #008000;
        }</style>  
<table id="T_4a476e71_9562_11e8_a1ed_525400123456" ><caption>Correlations between size measures</caption> 
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
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row0" class="row_heading level0 row0" >vmlinux</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col0" class="data row0 col0" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col1" class="data row0 col1" >0.879077</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col2" class="data row0 col2" >0.879127</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col3" class="data row0 col3" >0.877775</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col4" class="data row0 col4" >0.88717</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col5" class="data row0 col5" >0.886101</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col6" class="data row0 col6" >0.88593</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col7" class="data row0 col7" >0.868176</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col8" class="data row0 col8" >0.868241</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col9" class="data row0 col9" >0.866497</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col10" class="data row0 col10" >0.86861</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col11" class="data row0 col11" >0.86868</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col12" class="data row0 col12" >0.866788</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col13" class="data row0 col13" >0.892317</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col14" class="data row0 col14" >0.89236</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col15" class="data row0 col15" >0.891228</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col16" class="data row0 col16" >0.897058</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col17" class="data row0 col17" >0.897096</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row0_col18" class="data row0 col18" >0.895596</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row1" class="row_heading level0 row1" >GZIP-bzImage</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col0" class="data row1 col0" >0.879077</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col1" class="data row1 col1" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col2" class="data row1 col2" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col3" class="data row1 col3" >0.999988</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col4" class="data row1 col4" >0.999388</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col5" class="data row1 col5" >0.99792</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col6" class="data row1 col6" >0.999495</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col7" class="data row1 col7" >0.999331</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col8" class="data row1 col8" >0.999332</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col9" class="data row1 col9" >0.999276</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col10" class="data row1 col10" >0.99096</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col11" class="data row1 col11" >0.99096</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col12" class="data row1 col12" >0.990934</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col13" class="data row1 col13" >0.999335</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col14" class="data row1 col14" >0.999332</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col15" class="data row1 col15" >0.999413</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col16" class="data row1 col16" >0.998628</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col17" class="data row1 col17" >0.998624</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row1_col18" class="data row1 col18" >0.998387</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row2" class="row_heading level0 row2" >GZIP-vmlinux</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col0" class="data row2 col0" >0.879127</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col1" class="data row2 col1" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col2" class="data row2 col2" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col3" class="data row2 col3" >0.999987</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col4" class="data row2 col4" >0.999393</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col5" class="data row2 col5" >0.997925</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col6" class="data row2 col6" >0.999498</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col7" class="data row2 col7" >0.99933</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col8" class="data row2 col8" >0.999331</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col9" class="data row2 col9" >0.999274</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col10" class="data row2 col10" >0.990961</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col11" class="data row2 col11" >0.99096</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col12" class="data row2 col12" >0.990933</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col13" class="data row2 col13" >0.999339</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col14" class="data row2 col14" >0.999336</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col15" class="data row2 col15" >0.999416</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col16" class="data row2 col16" >0.998634</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col17" class="data row2 col17" >0.998629</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row2_col18" class="data row2 col18" >0.998391</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row3" class="row_heading level0 row3" >GZIP</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col0" class="data row3 col0" >0.877775</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col1" class="data row3 col1" >0.999988</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col2" class="data row3 col2" >0.999987</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col3" class="data row3 col3" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col4" class="data row3 col4" >0.99926</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col5" class="data row3 col5" >0.99779</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col6" class="data row3 col6" >0.999392</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col7" class="data row3 col7" >0.999347</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col8" class="data row3 col8" >0.999346</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col9" class="data row3 col9" >0.999322</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col10" class="data row3 col10" >0.990938</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col11" class="data row3 col11" >0.990936</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col12" class="data row3 col12" >0.990945</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col13" class="data row3 col13" >0.999225</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col14" class="data row3 col14" >0.999221</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col15" class="data row3 col15" >0.999325</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col16" class="data row3 col16" >0.99847</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col17" class="data row3 col17" >0.998465</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row3_col18" class="data row3 col18" >0.998248</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row4" class="row_heading level0 row4" >BZIP2-bzImage</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col0" class="data row4 col0" >0.88717</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col1" class="data row4 col1" >0.999388</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col2" class="data row4 col2" >0.999393</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col3" class="data row4 col3" >0.99926</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col4" class="data row4 col4" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col5" class="data row4 col5" >0.998538</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col6" class="data row4 col6" >0.999987</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col7" class="data row4 col7" >0.99854</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col8" class="data row4 col8" >0.998546</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col9" class="data row4 col9" >0.998339</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col10" class="data row4 col10" >0.989612</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col11" class="data row4 col11" >0.989618</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col12" class="data row4 col12" >0.989423</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col13" class="data row4 col13" >0.999569</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col14" class="data row4 col14" >0.99957</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col15" class="data row4 col15" >0.999543</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col16" class="data row4 col16" >0.999325</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col17" class="data row4 col17" >0.999324</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row4_col18" class="data row4 col18" >0.998989</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row5" class="row_heading level0 row5" >BZIP2-vmlinux</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col0" class="data row5 col0" >0.886101</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col1" class="data row5 col1" >0.99792</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col2" class="data row5 col2" >0.997925</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col3" class="data row5 col3" >0.99779</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col4" class="data row5 col4" >0.998538</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col5" class="data row5 col5" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col6" class="data row5 col6" >0.998524</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col7" class="data row5 col7" >0.997071</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col8" class="data row5 col8" >0.997078</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col9" class="data row5 col9" >0.996869</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col10" class="data row5 col10" >0.988067</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col11" class="data row5 col11" >0.988073</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col12" class="data row5 col12" >0.987875</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col13" class="data row5 col13" >0.99811</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col14" class="data row5 col14" >0.998111</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col15" class="data row5 col15" >0.998082</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col16" class="data row5 col16" >0.997871</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col17" class="data row5 col17" >0.99787</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row5_col18" class="data row5 col18" >0.997534</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row6" class="row_heading level0 row6" >BZIP2</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col0" class="data row6 col0" >0.88593</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col1" class="data row6 col1" >0.999495</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col2" class="data row6 col2" >0.999498</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col3" class="data row6 col3" >0.999392</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col4" class="data row6 col4" >0.999987</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col5" class="data row6 col5" >0.998524</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col6" class="data row6 col6" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col7" class="data row6 col7" >0.998675</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col8" class="data row6 col8" >0.99868</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col9" class="data row6 col9" >0.998505</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col10" class="data row6 col10" >0.989707</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col11" class="data row6 col11" >0.989711</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col12" class="data row6 col12" >0.989553</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col13" class="data row6 col13" >0.999575</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col14" class="data row6 col14" >0.999575</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col15" class="data row6 col15" >0.999571</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col16" class="data row6 col16" >0.99928</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col17" class="data row6 col17" >0.999279</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row6_col18" class="data row6 col18" >0.998966</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row7" class="row_heading level0 row7" >LZMA-bzImage</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col0" class="data row7 col0" >0.868176</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col1" class="data row7 col1" >0.999331</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col2" class="data row7 col2" >0.99933</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col3" class="data row7 col3" >0.999347</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col4" class="data row7 col4" >0.99854</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col5" class="data row7 col5" >0.997071</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col6" class="data row7 col6" >0.998675</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col7" class="data row7 col7" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col8" class="data row7 col8" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col9" class="data row7 col9" >0.999981</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col10" class="data row7 col10" >0.99032</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col11" class="data row7 col11" >0.990318</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col12" class="data row7 col12" >0.990332</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col13" class="data row7 col13" >0.997681</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col14" class="data row7 col14" >0.997677</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col15" class="data row7 col15" >0.997783</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col16" class="data row7 col16" >0.996613</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col17" class="data row7 col17" >0.996607</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row7_col18" class="data row7 col18" >0.996413</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row8" class="row_heading level0 row8" >LZMA-vmlinux</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col0" class="data row8 col0" >0.868241</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col1" class="data row8 col1" >0.999332</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col2" class="data row8 col2" >0.999331</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col3" class="data row8 col3" >0.999346</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col4" class="data row8 col4" >0.998546</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col5" class="data row8 col5" >0.997078</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col6" class="data row8 col6" >0.99868</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col7" class="data row8 col7" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col8" class="data row8 col8" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col9" class="data row8 col9" >0.999979</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col10" class="data row8 col10" >0.990322</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col11" class="data row8 col11" >0.99032</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col12" class="data row8 col12" >0.990332</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col13" class="data row8 col13" >0.997687</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col14" class="data row8 col14" >0.997683</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col15" class="data row8 col15" >0.997788</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col16" class="data row8 col16" >0.996621</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col17" class="data row8 col17" >0.996615</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row8_col18" class="data row8 col18" >0.99642</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row9" class="row_heading level0 row9" >LZMA</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col0" class="data row9 col0" >0.866497</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col1" class="data row9 col1" >0.999276</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col2" class="data row9 col2" >0.999274</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col3" class="data row9 col3" >0.999322</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col4" class="data row9 col4" >0.998339</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col5" class="data row9 col5" >0.996869</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col6" class="data row9 col6" >0.998505</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col7" class="data row9 col7" >0.999981</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col8" class="data row9 col8" >0.999979</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col9" class="data row9 col9" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col10" class="data row9 col10" >0.990252</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col11" class="data row9 col11" >0.990248</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col12" class="data row9 col12" >0.990307</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col13" class="data row9 col13" >0.997502</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col14" class="data row9 col14" >0.997497</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col15" class="data row9 col15" >0.997632</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col16" class="data row9 col16" >0.996373</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col17" class="data row9 col17" >0.996366</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row9_col18" class="data row9 col18" >0.996199</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row10" class="row_heading level0 row10" >XZ-bzImage</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col0" class="data row10 col0" >0.86861</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col1" class="data row10 col1" >0.99096</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col2" class="data row10 col2" >0.990961</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col3" class="data row10 col3" >0.990938</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col4" class="data row10 col4" >0.989612</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col5" class="data row10 col5" >0.988067</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col6" class="data row10 col6" >0.989707</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col7" class="data row10 col7" >0.99032</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col8" class="data row10 col8" >0.990322</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col9" class="data row10 col9" >0.990252</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col10" class="data row10 col10" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col11" class="data row10 col11" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col12" class="data row10 col12" >0.999976</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col13" class="data row10 col13" >0.990174</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col14" class="data row10 col14" >0.990171</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col15" class="data row10 col15" >0.990241</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col16" class="data row10 col16" >0.988915</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col17" class="data row10 col17" >0.988911</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row10_col18" class="data row10 col18" >0.988668</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row11" class="row_heading level0 row11" >XZ-vmlinux</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col0" class="data row11 col0" >0.86868</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col1" class="data row11 col1" >0.99096</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col2" class="data row11 col2" >0.99096</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col3" class="data row11 col3" >0.990936</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col4" class="data row11 col4" >0.989618</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col5" class="data row11 col5" >0.988073</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col6" class="data row11 col6" >0.989711</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col7" class="data row11 col7" >0.990318</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col8" class="data row11 col8" >0.99032</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col9" class="data row11 col9" >0.990248</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col10" class="data row11 col10" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col11" class="data row11 col11" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col12" class="data row11 col12" >0.999974</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col13" class="data row11 col13" >0.990178</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col14" class="data row11 col14" >0.990176</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col15" class="data row11 col15" >0.990245</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col16" class="data row11 col16" >0.988923</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col17" class="data row11 col17" >0.988919</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row11_col18" class="data row11 col18" >0.988674</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row12" class="row_heading level0 row12" >XZ</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col0" class="data row12 col0" >0.866788</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col1" class="data row12 col1" >0.990934</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col2" class="data row12 col2" >0.990933</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col3" class="data row12 col3" >0.990945</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col4" class="data row12 col4" >0.989423</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col5" class="data row12 col5" >0.987875</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col6" class="data row12 col6" >0.989553</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col7" class="data row12 col7" >0.990332</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col8" class="data row12 col8" >0.990332</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col9" class="data row12 col9" >0.990307</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col10" class="data row12 col10" >0.999976</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col11" class="data row12 col11" >0.999974</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col12" class="data row12 col12" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col13" class="data row12 col13" >0.990011</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col14" class="data row12 col14" >0.990007</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col15" class="data row12 col15" >0.990109</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col16" class="data row12 col16" >0.988685</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col17" class="data row12 col17" >0.98868</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row12_col18" class="data row12 col18" >0.988465</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row13" class="row_heading level0 row13" >LZO-bzImage</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col0" class="data row13 col0" >0.892317</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col1" class="data row13 col1" >0.999335</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col2" class="data row13 col2" >0.999339</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col3" class="data row13 col3" >0.999225</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col4" class="data row13 col4" >0.999569</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col5" class="data row13 col5" >0.99811</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col6" class="data row13 col6" >0.999575</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col7" class="data row13 col7" >0.997681</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col8" class="data row13 col8" >0.997687</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col9" class="data row13 col9" >0.997502</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col10" class="data row13 col10" >0.990174</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col11" class="data row13 col11" >0.990178</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col12" class="data row13 col12" >0.990011</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col13" class="data row13 col13" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col14" class="data row13 col14" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col15" class="data row13 col15" >0.99999</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col16" class="data row13 col16" >0.999855</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col17" class="data row13 col17" >0.999854</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row13_col18" class="data row13 col18" >0.999525</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row14" class="row_heading level0 row14" >LZO-vmlinux</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col0" class="data row14 col0" >0.89236</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col1" class="data row14 col1" >0.999332</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col2" class="data row14 col2" >0.999336</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col3" class="data row14 col3" >0.999221</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col4" class="data row14 col4" >0.99957</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col5" class="data row14 col5" >0.998111</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col6" class="data row14 col6" >0.999575</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col7" class="data row14 col7" >0.997677</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col8" class="data row14 col8" >0.997683</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col9" class="data row14 col9" >0.997497</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col10" class="data row14 col10" >0.990171</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col11" class="data row14 col11" >0.990176</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col12" class="data row14 col12" >0.990007</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col13" class="data row14 col13" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col14" class="data row14 col14" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col15" class="data row14 col15" >0.999989</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col16" class="data row14 col16" >0.999857</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col17" class="data row14 col17" >0.999855</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row14_col18" class="data row14 col18" >0.999526</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row15" class="row_heading level0 row15" >LZO</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col0" class="data row15 col0" >0.891228</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col1" class="data row15 col1" >0.999413</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col2" class="data row15 col2" >0.999416</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col3" class="data row15 col3" >0.999325</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col4" class="data row15 col4" >0.999543</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col5" class="data row15 col5" >0.998082</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col6" class="data row15 col6" >0.999571</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col7" class="data row15 col7" >0.997783</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col8" class="data row15 col8" >0.997788</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col9" class="data row15 col9" >0.997632</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col10" class="data row15 col10" >0.990241</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col11" class="data row15 col11" >0.990245</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col12" class="data row15 col12" >0.990109</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col13" class="data row15 col13" >0.99999</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col14" class="data row15 col14" >0.999989</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col15" class="data row15 col15" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col16" class="data row15 col16" >0.999802</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col17" class="data row15 col17" >0.999799</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row15_col18" class="data row15 col18" >0.99949</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row16" class="row_heading level0 row16" >LZ4-bzImage</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col0" class="data row16 col0" >0.897058</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col1" class="data row16 col1" >0.998628</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col2" class="data row16 col2" >0.998634</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col3" class="data row16 col3" >0.99847</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col4" class="data row16 col4" >0.999325</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col5" class="data row16 col5" >0.997871</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col6" class="data row16 col6" >0.99928</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col7" class="data row16 col7" >0.996613</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col8" class="data row16 col8" >0.996621</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col9" class="data row16 col9" >0.996373</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col10" class="data row16 col10" >0.988915</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col11" class="data row16 col11" >0.988923</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col12" class="data row16 col12" >0.988685</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col13" class="data row16 col13" >0.999855</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col14" class="data row16 col14" >0.999857</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col15" class="data row16 col15" >0.999802</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col16" class="data row16 col16" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col17" class="data row16 col17" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row16_col18" class="data row16 col18" >0.999673</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row17" class="row_heading level0 row17" >LZ4-vmlinux</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col0" class="data row17 col0" >0.897096</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col1" class="data row17 col1" >0.998624</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col2" class="data row17 col2" >0.998629</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col3" class="data row17 col3" >0.998465</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col4" class="data row17 col4" >0.999324</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col5" class="data row17 col5" >0.99787</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col6" class="data row17 col6" >0.999279</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col7" class="data row17 col7" >0.996607</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col8" class="data row17 col8" >0.996615</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col9" class="data row17 col9" >0.996366</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col10" class="data row17 col10" >0.988911</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col11" class="data row17 col11" >0.988919</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col12" class="data row17 col12" >0.98868</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col13" class="data row17 col13" >0.999854</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col14" class="data row17 col14" >0.999855</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col15" class="data row17 col15" >0.999799</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col16" class="data row17 col16" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col17" class="data row17 col17" >1</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row17_col18" class="data row17 col18" >0.999672</td> 
    </tr>    <tr> 
        <th id="T_4a476e71_9562_11e8_a1ed_525400123456level0_row18" class="row_heading level0 row18" >LZ4</th> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col0" class="data row18 col0" >0.895596</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col1" class="data row18 col1" >0.998387</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col2" class="data row18 col2" >0.998391</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col3" class="data row18 col3" >0.998248</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col4" class="data row18 col4" >0.998989</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col5" class="data row18 col5" >0.997534</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col6" class="data row18 col6" >0.998966</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col7" class="data row18 col7" >0.996413</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col8" class="data row18 col8" >0.99642</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col9" class="data row18 col9" >0.996199</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col10" class="data row18 col10" >0.988668</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col11" class="data row18 col11" >0.988674</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col12" class="data row18 col12" >0.988465</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col13" class="data row18 col13" >0.999525</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col14" class="data row18 col14" >0.999526</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col15" class="data row18 col15" >0.99949</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col16" class="data row18 col16" >0.999673</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col17" class="data row18 col17" >0.999672</td> 
        <td id="T_4a476e71_9562_11e8_a1ed_525400123456row18_col18" class="data row18 col18" >1</td> 
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
    Int64Index: 12079 entries, 0 to 12499
    Columns: 9317 entries, X86_LOCAL_APIC to LZ4
    dtypes: int64(9317)
    memory usage: 858.7 MB





    ((12079, 9317), None)




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
    1. feature UBSAN_SANITIZE_ALL 2018 (0.069361)
    2. feature X86_NEED_RELOCS 4348 (0.048769)
    3. feature KCOV_INSTRUMENT_ALL 6414 (0.044007)
    4. feature MAXSMP 8324 (0.035185)
    5. feature RANDOMIZE_BASE 4343 (0.034149)
    6. feature X86_VSMP 4584 (0.032832)
    7. feature LOCK_STAT 7445 (0.026078)
    8. feature XFS_DEBUG 4411 (0.025647)
    9. feature USB 3169 (0.018133)
    10. feature MODULES 5249 (0.017434)
    11. feature STRICT_MODULE_RWX 9231 (0.014454)
    12. feature DYNAMIC_DEBUG 5300 (0.012523)
    13. feature EXPERT 1375 (0.011780)
    14. feature KALLSYMS_ABSOLUTE_PERCPU 6150 (0.011554)
    15. feature DRM_I915 3607 (0.011409)
    16. feature GENERIC_TRACER 8579 (0.010924)
    17. feature PRINTK_NMI 6173 (0.010599)
    18. feature DRM_AMDGPU 2826 (0.010435)
    19. feature SND_SOC_ARIZONA 7867 (0.010418)
    20. feature BLK_MQ_PCI 2406 (0.010402)
    21. feature PRINTK 3396 (0.009910)
    22. feature ACPI_VIDEO 6511 (0.009781)
    23. feature XFS_FS 4103 (0.009422)
    24. feature MDIO 7226 (0.009383)
    25. feature CONTEXT_SWITCH_TRACER 8572 (0.009148)
    26. feature RAID6_PQ 998 (0.009090)
    27. feature MAC80211 7605 (0.008624)
    28. feature DRM_RADEON 2803 (0.008131)
    29. feature SND_SOC_HDMI_CODEC 2469 (0.007297)
    30. feature CFG80211 8880 (0.006956)
    31. feature DRM_NOUVEAU 2447 (0.006875)
    32. feature SCSI_ISCSI_ATTRS 227 (0.006748)
    33. feature SND_SOC_RT5640 8397 (0.006655)
    34. feature NOP_TRACER 8562 (0.006510)
    35. feature DST_CACHE 1292 (0.006455)
    36. feature PINCTRL_MTK 1651 (0.006432)
    37. feature INET 550 (0.006119)
    38. feature SND_HDA_CORE 9070 (0.005739)
    39. feature JBD2 3624 (0.005713)
    40. feature HOTPLUG_CPU 4837 (0.005538)
    41. feature V4L2_FWNODE 4149 (0.005515)
    42. feature SND_HWDEP 3477 (0.005465)
    43. feature SERIAL_MCTRL_GPIO 4038 (0.005325)
    44. feature TRACEPOINTS 7604 (0.005318)
    45. feature AD5933 723 (0.005232)
    46. feature NETFILTER_XTABLES 6389 (0.005232)
    47. feature FB_SVGALIB 202 (0.004928)
    48. feature I2C 4638 (0.004775)
    49. feature SCSI_LOWLEVEL 259 (0.004468)
    50. feature DEBUG_LOCK_ALLOC 6881 (0.004458)
    51. feature BLK_SCSI_REQUEST 822 (0.004413)
    52. feature MMU_NOTIFIER 1814 (0.004360)
    53. feature GENERIC_NET_UTILS 150 (0.004236)
    54. feature OCFS2_FS 4434 (0.004162)
    55. feature FB_DDC 189 (0.004099)
    56. feature INTEL_GTT 1174 (0.004051)
    57. feature GPIO_ACPI 2097 (0.003741)
    58. feature NEED_MULTIPLE_NODES 676 (0.003676)
    59. feature DRM_TTM 2419 (0.003660)
    60. feature XPS 564 (0.003619)
    61. feature DVB_STV0900 1630 (0.003607)
    62. feature SND_SOC_WM5110 3518 (0.003592)
    63. feature INFINIBAND 928 (0.003578)
    64. feature ARMADA375_USBCLUSTER_PHY 7978 (0.003481)
    65. feature SG_POOL 4021 (0.003453)
    66. feature CEPH_LIB 8756 (0.003311)
    67. feature DRM_MIPI_DSI 1982 (0.003242)
    68. feature KALLSYMS_ALL 6145 (0.003231)
    69. feature SND_SOC_CS4270 6191 (0.003148)
    70. feature VIDEOBUF2_VMALLOC 8627 (0.003121)
    71. feature PM_GENERIC_DOMAINS_OF 5656 (0.003101)
    72. feature SND_SOC_WM9081 3479 (0.003010)
    73. feature SND_SOC_WM1250_EV1 3492 (0.002952)
    74. feature FUNCTION_TRACER 1172 (0.002920)
    75. feature MII 3853 (0.002919)
    76. feature BTRFS_FS 4485 (0.002918)
    77. feature NET_VENDOR_MELLANOX 6678 (0.002918)
    78. feature UBSAN_ALIGNMENT 2024 (0.002902)
    79. feature USB_HID 3254 (0.002889)
    80. feature VIDEO_TVEEPROM 7942 (0.002884)
    81. feature VIDEOBUF_GEN 4152 (0.002870)
    82. feature ACPI_BUTTON 6017 (0.002856)
    83. feature SCSI_SAS_ATTRS 9150 (0.002829)
    84. feature CPUMASK_OFFSTACK 8325 (0.002742)
    85. feature SND_SOC_WM8904 3195 (0.002712)
    86. feature MTD_BLKDEVS 3158 (0.002631)
    87. feature VIDEOBUF_VMALLOC 4159 (0.002608)
    88. feature BT_INTEL 5698 (0.002581)
    89. feature PRIME_NUMBERS 2006 (0.002565)
    90. feature RAID_ATTRS 9095 (0.002564)
    91. feature DQL 603 (0.002544)
    92. feature HIBERNATE_CALLBACKS 6042 (0.002482)
    93. feature VIDEOBUF2_DMA_SG 2542 (0.002425)
    94. feature MDIO_CAVIUM 5004 (0.002364)
    95. feature IP_SCTP 9073 (0.002363)
    96. feature HAVE_NET_DSA 5138 (0.002360)
    97. feature SND_USB_LINE6 1664 (0.002352)
    98. feature AF_RXRPC 6633 (0.002332)
    99. feature VIDEO_MSP3400 7974 (0.002330)
    100. feature NTB_TOOL 3531 (0.002320)
    Prediction score (MAE): 2.07
    Prediction score (MSE): 13741330.94
    Prediction score (R2): 0.92
    Prediction score (MRE): 6.98



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
    Int64Index: 12079 entries, 0 to 12499
    Columns: 9317 entries, X86_LOCAL_APIC to LZ4
    dtypes: int64(9317)
    memory usage: 858.7 MB



```python
rawtuxdata['vmlinux'].sort_values()
```




    8035       7317008
    2269      10572520
    1672      10857304
    8036      10865408
    10798     10889536
    12037     10892608
    2959      10918504
    12267     10962488
    3620      10981392
    57        10992912
    8277      11035128
    9253      11105984
    10778     11116560
    2963      11117600
    5607      11129920
    8485      11148984
    5492      11161272
    27        11170448
    6981      11191840
    12080     11193720
    1795      11200416
    6552      11237448
    8953      11295064
    7567      11299728
    9901      11317488
    6598      11317608
    8220      11319536
    9817      11395976
    9535      11398144
    9699      11463608
               ...    
    2048     102586720
    261      104562552
    9289     105527560
    10373    107293136
    8380     107552944
    9224     108359616
    703      108475344
    8759     109415568
    149      109714704
    3564     109752480
    167      111089688
    285      113105856
    247      114508480
    8008     117212200
    136      117691960
    242      120413584
    5719     121892600
    220      123554520
    300      128829176
    12190    129896344
    507      140729592
    48       142625592
    9977     148573672
    148      154263088
    169      156060312
    241      170588104
    397      220759184
    385      236043464
    97       249380672
    215      250911288
    Name: vmlinux, Length: 12079, dtype: int64


