import matplotlib.pyplot as plt
import numpy as np
from skimage import data, img_as_float, io, segmentation, color
#from skimage.segmentation import chan_vese
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage.future import graph
import SimpleITK as sitk

img = sitk.ReadImage('D:\\dataset\\xVertSeg.v1\\Data1\\images\\image011.mhd')
imgArr = sitk.GetArrayFromImage(img)
org = np.array(list(reversed(img.GetOrigin())))
spacing = np.array(list(reversed(img.GetSpacing())))
view = sitk.GetArrayViewFromImage(img)[0, :, :]
arr = sitk.GetArrayFromImage(img)[0, :, :]
#arr = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
#print(arr)
plt.imshow(arr, cmap="gray")
plt.show()
#exit()
#Watershed

'''Input Image'''
img = arr
#img=img[:,20:-20]

'''Super Pixel Segmentation method'''
labels = segmentation.slic(img, compactness=10, n_segments=1024)
#labels = segmentation.quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
#gradient = sobel(rgb2gray(img))
#labels = segmentation.watershed(gradient, markers=250, compactness=0.001)

'''Region Adjacency Graph'''
g = graph.rag_mean_color(img, labels, mode='similarity')

'''Output Image'''
img_out = color.label2rgb(labels, img, kind='avg')
img_out_bound = segmentation.mark_boundaries(img_out, labels, (255,255,255))

'''Display Image and RAG'''
plt.imshow(img_out, cmap="gray")
plt.show()
#lg = graph.show_rag(labels, g, img)
#plt.colorbar(lg)
#plt.show()

#plt.imshow(img_out_bound)
#plt.show()