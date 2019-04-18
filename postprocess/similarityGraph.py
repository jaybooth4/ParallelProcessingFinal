import plotly.plotly as py
import plotly.offline as offline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import spatial

def similarityGraph(typeName, names, outputDir, data=None, inputDir=None):
    if data is None:
        data = pd.read_csv(inputDir + typeName + 'Distances.csv')
    if typeName == 'Styles':
        xData = data[['beer_style']].values.flatten()
    elif typeName == 'Beers':
        xData = data[['beer_name']].values.flatten()

    # Inverse of distance so that closer bars are higher
    yData = list(map(lambda dist: 1.0 / dist, data[['distance']].values.flatten()))
    data = [go.Bar(
        x=xData[:20],
        y=yData[:20]
    )]
    layout = go.Layout(
        title=('Most Smiliar ' + typeName + ' to ' + names),
        xaxis=dict(
            title=typeName,
            # ticklen=5,
            # zeroline=False,
            # gridwidth=2
        ),
        yaxis=dict(
            title='Similarity',
            # ticklen=5,
            # gridwidth=2,
        ),
        margin = dict(
            b = 150
        ),
        showlegend=False
    )
    fig = go.Figure(data=data, layout=layout)
    offline.plot(fig, filename=outputDir + typeName + 'Similarity.html')
