import matplotlib.pyplot as plt
from io import BytesIO
import base64

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def get_plot(plots, params):
    plt.switch_backend('AGG')
    plt.figure(figsize=(10,5))
    for index in plots:
       plt.plot(plots[index]['x'], plots[index]['y'])
    plt.xticks(rotation=25, horizontalalignment='right')
    plt.title(params['title'])
    plt.xlabel(params['xlabel'])
    plt.ylabel(params['ylabel'])
    plt.grid()
    if len(params['legend']) > 0:
       plt.legend(params['legend'])
    plt.tight_layout()
    graph = get_graph()
    return graph