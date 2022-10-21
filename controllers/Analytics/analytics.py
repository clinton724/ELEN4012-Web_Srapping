dict = {'plot-1': {
            'x':'john', 
            'y':'kani'},
        'plot-2': {
            'x': 'romeo',
            'y': 'ron'}
        }

params = {'title': 'this graph',
          'xlabel': 'price',
          'ylabel': 'time',
          'legend': ['one', 'two']}
print(params['legend'])
for k in dict:
    print(dict[k]['x'])

print(len([]))