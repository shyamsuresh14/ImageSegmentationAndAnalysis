from skimage import data, io, segmentation, color
from skimage.future import graph
import numpy as np
import cv2

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
img = io.imread("C:\\Users\\shyam\\Downloads\\quiz_bg2.png")
img = cv2.resize(img, (1024, 1024))
labels = segmentation.slic(img, compactness=30, n_segments=400)
g = graph.rag_mean_color(img, labels)
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

for (i, segVal) in enumerate(np.unique(new_labels)):
    print(i, segVal)
    mask = np.zeros(img_out.shape[:2], dtype="uint8")
    mask[new_labels == segVal] = 255
    print(mask);break
    '''cv2.imshow("Mask", mask)
    cv2.imshow("Applied", cv2.bitwise_and(img_out, img_out, mask = mask))
    cv2.waitKey(0)'''