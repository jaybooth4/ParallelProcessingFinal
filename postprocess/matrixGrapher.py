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

OUTPUT = '../output/'
BEERVECTORS = "testMatrix.csv"

class MatrixGrapher:
    
    def __init__(self, graphType='beer'):    
        vectors = pd.read_csv(BEERVECTORS)
        self.data = vectors.iloc[:, 1:].values.tolist()
        self.ids = vectors.iloc[:, 0]
        self.graphType = graphType
    
    def graphPCA(self):
        output_file(OUTPUT + "pca" + ".html")
        
        pca = PCA(n_components=2)
        df = pd.DataFrame(pca.fit_transform(self.data), columns=['PCA1', 'PCA2'])

        source = ColumnDataSource(
            data = dict(
                PCA1 = df['PCA1'],
                PCA2 = df['PCA2'],
                colors = [all_palettes['Category20'][20][i % 20] for i in self.ids],
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


    # def graphLDA(self, name):
    #     output_file("results/kmeans-clustering-" + name + ".html")
        
    #     pca = PCA(n_components=2)
    #     df = pd.DataFrame(pca.fit_transform(self.docs_rep), columns=['PCA1', 'PCA2'])

    #     source = ColumnDataSource(
    #         data = dict(
    #             PCA1 = df['PCA1'],
    #             PCA2 = df['PCA2'],
    #             colors = [all_palettes['Category20'][20][i] for i in self.kmeans_labels],
    #             alpha = [0.9] * self.num_docs,
    #             size = [7] * self.num_docs
    #         )
    #     )

    #     plot = figure(title="K-Means Clustering")
    #     plot.circle('PCA1', 
    #                 'PCA2', 
    #                 fill_color='colors',
    #                 alpha='alpha',
    #                 size='size',
    #                 source=source
    #                )

    #     layout = column(plot)
    #     show(layout)

    def graphTSNE(self, perplexity=20):
        ''' Runs t-SNE on vector representations, then graphs groupings ''' 
        output_file(OUTPUT + self.graphType + '-tsne.html')
        tsne = TSNE(perplexity=20)
        tsne_embedding = tsne.fit_transform(self.data)
        tsne_embedding = pd.DataFrame(tsne_embedding, columns=['x','y'])
        # tsne_embedding['hue'] = self.data.argmax(axis=1)

        source = ColumnDataSource(
            data=dict(
                x = tsne_embedding.x,
                y = tsne_embedding.y,
                colors = [all_palettes['Category20'][20][i % 20] for i in self.ids],
                alpha = [0.9] * tsne_embedding.shape[0],
                size = [7] * tsne_embedding.shape[0]
            )
        )

        hover = HoverTool(
            tooltips = [
                ("Document", "@doc_description")
            ]
        )

        plot = figure(title="Graph of " + self.graphType + " embeddings",
                        tools=[hover])
        plot.circle('x', 'y', size='size', fill_color='colors', 
                        alpha='alpha', line_alpha=0, line_width=0.01, source=source, name="TSNE")

        layout = column(plot)
        show(layout)
