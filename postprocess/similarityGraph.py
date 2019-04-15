import plotly.plotly as py
import plotly.offline as offline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import spatial

INPUT = '../output/'
OUTPUT = '../output/'

def similarityGraph(typeName):
    data = pd.read_csv(INPUT + typeName + 'Distances.csv')
    if typeName == 'style':
        xData = data[['beer_style']].values.flatten()
    # Inverse of distance so that closer bars are higher
    yData = list(map(lambda dist: 1.0 / dist, data[['distance']].values.flatten()))
    data = [go.Bar(
        x=xData,
        y=yData
    )]
    layout = go.Layout(
        title=('Most Smiliar ' + typeName + '\'s'),
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
        showlegend=False
    )
    fig = go.Figure(data=data, layout=layout)
    offline.plot(fig, filename=OUTPUT + typeName + 'Similarity.html')
