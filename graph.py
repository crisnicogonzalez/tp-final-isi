import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


df = pd.read_csv('ar_properties.csv')
df = df.loc[df['l2'] == 'Capital Federal']
df = df.loc[df['operation_type'] == 'Venta']
df = df.loc[df['property_type'] == 'Departamento']
df = df.loc[df['currency'] == 'USD']
df['created_on'] = pd.to_datetime(df['created_on'])
df['year'] = df['created_on'].map(lambda x:x.year)



ord_enc = OrdinalEncoder()
enc_df = pd.DataFrame(ord_enc.fit_transform(df), columns=list(df.columns))
categories = pd.DataFrame(np.array(ord_enc.categories_).transpose(), columns=list(df.columns))

# Generate the random noise
xnoise, ynoise = np.random.random(len(df))/2, np.random.random(len(df))/2 # The noise is in the range 0 to 0.5

# Plot the scatterplot
plt.scatter(enc_df["rooms"]+xnoise, enc_df["target"]+ynoise, alpha=0.5)
# You can also set xticks and yticks to be your category names:
plt.xticks([0.25, 1.25, 2.25], categories["rooms"]) # The reason the xticks start at 0.25
# and go up in increments of 1 is because the center of the noise will be around 0.25 and ordinal
# encoded labels go up in increments of 1.
plt.yticks([0.25, 1.25, 2.25], categories["target"]) # This has the same reason explained for xticks

# Extra unnecessary styling...
plt.grid()
sns.despine(left=True, bottom=True)