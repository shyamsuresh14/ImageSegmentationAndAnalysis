from skimage import data, io, segmentation, color
from skimage.future import graph
import numpy as np
import cv2
import matplotlib.pyplot as plt

def weight_mean_color(graph, src, dst, n):
    d = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    d = np.linalg.norm(d)
    return {'weight': d}
def merge_mean_color(graph, src, dst):
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] += graph.node[dst]['total color'] / graph.node[dst]['pixel count']

#main
#img = data.coffee()
img = io.imread("images/brain_mri.jpg")
img = cv2.resize(img, (1024, 1024))
labels = segmentation.slic(img, compactness=30, n_segments=400)
g = graph.rag_mean_color(img, labels, mode='similarity')

#x = plt.figure(1)
lg = graph.show_rag(labels, g, img)
plt.colorbar(lg)
#plt.show()

new_labels = graph.cut_normalized(labels, g)
#print(new_labels)

new_g = graph.rag_mean_color(img, new_labels)
lg = graph.show_rag(new_labels, new_g, img)
plt.colorbar(lg)
plt.show()

img_out = color.label2rgb(new_labels, img, kind='avg')
#img_out = segmentation.mark_boundaries(img_out, new_labels, (0,0,0))
io.imshow(img_out)
io.show()
#print(g)
new_labels = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False, in_place_merge=True,
                                      merge_func=merge_mean_color, weight_func=weight_mean_color)
print(new_labels)
img_out = color.label2rgb(new_labels, img, kind='avg')
img_out = segmentation.mark_boundaries(img_out, new_labels, (0,0,0))
io.imshow(img)
io.show()
io.imshow(img_out)
io.show()