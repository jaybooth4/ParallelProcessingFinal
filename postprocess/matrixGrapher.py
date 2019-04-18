import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.io import output_file
from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, CustomJS, HoverTool, Slider
from bokeh.palettes import all_palettes
from bokeh.plotting import figure, show
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from util import readMultipartCSV, getBeerIdsByStyles

class MatrixGrapher:

    def __init__(self, beerVectorDir, beerLookup, outputDir, styles, graphType='beer'):
        vectors = readMultipartCSV(beerVectorDir)
        idsForStyles, idToStyleMap = getBeerIdsByStyles(styles, beerLookup)
        vectors = vectors.loc[vectors[0].isin(idsForStyles)]
        self.outputDir = outputDir
        self.data = vectors.iloc[:, 1:].values.tolist()
        self.ids = list(map(lambda elem: styles.index(idToStyleMap[elem]), vectors.iloc[:, 0]))
        self.graphType = graphType
    
    def graphPCA(self):
        output_file(self.outputDir + "pca" + ".html")
        
        pca = PCA(n_components=2)
        df = pd.DataFrame(pca.fit_transform(self.data), columns=['PCA1', 'PCA2'])

        source = ColumnDataSource(
            data = dict(
                PCA1 = df['PCA1'],
                PCA2 = df['PCA2'],
                colors = [all_palettes['Category20'][20][i] for i in self.ids],
                alpha = [0.9] * len(self.ids),
                size = [7] * len(self.ids)
            )
        )

        plot = figure(title="PCA for " + self.graphType + " data")
        plot.circle('PCA1', 
                    'PCA2', 
                    fill_color='colors',
                    alpha='alpha',
                    size='size',
                    source=source
                   )

        layout = column(plot)
        show(layout)

    def graphTSNE(self, perplexity=20):
        ''' Runs t-SNE on vector representations, then graphs groupings ''' 
        output_file(self.outputDir + self.graphType + '-tsne.html')
        tsne = TSNE(perplexity=perplexity)
        tsne_embedding = tsne.fit_transform(self.data)
        tsne_embedding = pd.DataFrame(tsne_embedding, columns=['x','y'])
        # tsne_embedding['hue'] = self.data.argmax(axis=1)

        source = ColumnDataSource(
            data=dict(
                x = tsne_embedding.x,
                y = tsne_embedding.y,
                colors = [all_palettes['Category20'][20][i] for i in self.ids],
                alpha = [0.9] * tsne_embedding.shape[0],
                size = [7] * tsne_embedding.shape[0]
            )
        )

        plot = figure(title="Graph of " + self.graphType + " embeddings")
        plot.circle('x', 'y', size='size', fill_color='colors', 
                        alpha='alpha', line_alpha=0, line_width=0.01, source=source, name="TSNE")

        layout = column(plot)
        show(layout)
