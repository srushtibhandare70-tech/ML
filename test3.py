import pandas as pd
import pandas as np
dict={'first score':[100,90,np.nan,95],
      'second score':[np.nan,40,80,98]
      'thired score':[np.nan,40,80,98],
      }
df=pd.DataFrame(dict)
df.isnull()

