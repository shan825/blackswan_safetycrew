import pandas as pd
import numpy as np
import plotly.express as px

# Used results from Doug's Model on 9-19-2022
df = pd.read_csv("model_results\PyTorch_800caseResults__09-19-2022_163431.csv")
print(df.head())

fig = px.scatter_3d(df, 
                    x='xTemp', 
                    y='yVol', 
                    z='Target',
                    color='CorrectPred')
fig.write_html("visualizations/3d_scatter_animated.html")
fig.show()
