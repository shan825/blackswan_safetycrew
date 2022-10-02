import pandas as pd
import numpy as np
import plotly.express as px

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)

# Used results from Doug's Model on 9-19-2022
df = pd.read_csv("model_results\PyTorch_800caseResults__09-19-2022_163431.csv")
#print(df.head())

# Original 3D scatterplot code

#Create X,Y, and Z variables
#z = df['Target']
#x = df['xTemp']
#y = df['yVol']

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

#for s in df['CorrectPred'].unique():
#    ax.scatter3D(df.xTemp[df['CorrectPred']==s],
#               df.yVol[df['CorrectPred']==s],
#               df.Target[df['CorrectPred']==s],label=s)
#plt.title("3D Scatter of Temp Vol and Target")
#plt.xlabel('Temperature')
#plt.ylabel('Volume')
#ax.set_zlabel('Target')    
#ax.legend()
#fig.show()

# Use Plotly to create interactive 3D scatterplot of holdout
fig = px.scatter_3d(df, 
                    x='xTemp', 
                    y='yVol', 
                    z='Target',
                    color='CorrectPred')
fig.write_html("visualizations/3d_scatter_animated.html")
fig.show()
