# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 12:14:24 2021

@author: alfah
"""
#import libraries 
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from glob import glob

path = 'seg_samples'
#%%
#K-means, K=3
#0-eosinophil, 2497-lymphocite, 4980-monocyte, 7458-neutrophil
#change filenames depending on which image you want to segment
img = cv2.imread(path+'/BloodImage_00029.jpg')
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
'''
(T, threshInv) = cv2.threshold(img, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
'''

# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1,3))
 
# Convert to float type
pixel_vals = np.float32(pixel_vals)

#the below line of code defines the criteria for the algorithm to stop running,
#which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
#becomes 85%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
 
# then perform k-means clustering wit h number of clusters defined as 3
#also random centres are initially choosed for k-means clustering
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
 
# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
 
# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
blue = segmented_image[:,:,2]
#%%
#K-means - thresholding and saving the raw output
b_max = np.max(blue)
b_min = np.min(blue)
b = blue
b[b==b_max] = 255
b[b==b_min] = 0
#change filenames depending on which image you want to segment
cv2.imwrite(path+'/Kmeans_mid_29.jpg', np.uint8(b))
#%%
#using raw output to segment WBC from original image and saving the result
b_max = np.max(blue)
blue[blue<b_max] = 0
blue[blue == b_max] = 1
Blue = np.stack([blue, blue, blue], axis=2)
final = image*Blue
plt.imshow(final)
final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
#change filenames depending on which image you want to segment
cv2.imwrite(path+'/Kmeans_29.jpg', final)
#%%
#Manual Graph Cuts method
#1st - mark the foreground region using the mouse cursor and then press "t"
#2nd - mark the background region using the mouse cursor and rpess "g" to start the graph cuts operation
import cv2
import numpy as np
import maxflow
import matplotlib.pyplot as plt


class GraphMaker:

    foreground = 1
    background = 0

    seeds = 0
    segmented = 1

    default = 0.5
    MAXIMUM = 1000000000

    def __init__(self,filename):
        self.image = None
        self.graph = None
        self.overlay = None
        self.seed_overlay = None
        self.segment_overlay = None
        self.mask = None
        self.load_image(filename)
        self.background_seeds = []
        self.foreground_seeds = []
        self.background_average = np.array(3)
        self.foreground_average = np.array(3)
        self.nodes = []
        self.edges = []
        self.current_overlay = self.seeds

    def load_image(self, filename):
        self.image = cv2.imread(filename)
        self.graph = np.zeros_like(self.image)
        self.seed_overlay = np.zeros_like(self.image)
        self.segment_overlay = np.zeros_like(self.image)
        self.mask = None

    def add_seed(self, x, y, type):
        if self.image is None:
            print ('Please load an image before adding seeds.')
        if type == self.background:
            if not self.background_seeds.__contains__((x, y)):
                self.background_seeds.append((x, y))
                cv2.rectangle(self.seed_overlay, (x-1, y-1), (x+1, y+1), (0, 0, 255), -1)
        elif type == self.foreground:
            if not self.foreground_seeds.__contains__((x, y)):
                self.foreground_seeds.append((x, y))
                cv2.rectangle(self.seed_overlay, (x-1, y-1), (x+1, y+1), (0, 255, 0), -1)

    def clear_seeds(self):
        self.background_seeds = []
        self.foreground_seeds = []
        self.seed_overlay = np.zeros_like(self.seed_overlay)

    def get_overlay(self):
        if self.current_overlay == self.seeds:
            return self.seed_overlay
        else:
            return self.segment_overlay

    def get_image_with_overlay(self, overlayNumber):
        if overlayNumber == self.seeds:
            return cv2.addWeighted(self.image, 0.9, self.seed_overlay, 0.4, 0.1)
        else:
            return cv2.addWeighted(self.image, 0.9, self.segment_overlay, 0.4, 0.1)

    def create_graph(self):
        if len(self.background_seeds) == 0 or len(self.foreground_seeds) == 0:
            print ("Please enter at least one foreground and background seed.")
            return

        print ("Making graph")
        print ("Finding foreground and background averages")
        self.find_averages()

        print ("Populating nodes and edges")
        self.populate_graph()

        print ("Cutting graph")
        self.cut_graph()
        print ("finished cutting")
        return(self.save_image('D:\\ED6001_Assignment3\\48\\train_10Segmented.bmp'))

    def find_averages(self):
        self.graph = np.zeros((self.image.shape[0], self.image.shape[1]))
        print (self.graph.shape)
        self.graph.fill(self.default)
        self.background_average = np.zeros(3)
        self.foreground_average = np.zeros(3)

        for coordinate in self.background_seeds:
            self.graph[coordinate[1] - 1, coordinate[0] - 1] = 0
            #self.background_average += self.image[coordinate[1], coordinate[0]]

        #self.background_average /= len(self.background_seeds)

        for coordinate in self.foreground_seeds:
            self.graph[coordinate[1] - 1, coordinate[0] - 1] = 1
            #self.foreground_average += self.image[coordinate[1], coordinate[0]]

        #self.foreground_average /= len(self.foreground_seeds)

    def populate_graph(self):
        self.nodes = []
        self.edges = []

        # make all s and t connections for the graph
        for (y, x), value in np.ndenumerate(self.graph):
            # this is a background pixel
            if value == 0.0:
                self.nodes.append((self.get_node_num(x, y, self.image.shape), self.MAXIMUM, 0))

            # this is a foreground node
            elif value == 1.0:
                self.nodes.append((self.get_node_num(x, y, self.image.shape), 0, self.MAXIMUM))

            else:
                '''d_f = np.power(self.image[y, x] - self.foreground_average, 2)
                d_b = np.power(self.image[y, x] - self.background_average, 2)
                d_f = np.sum(d_f)
                d_b = np.sum(d_b)
                e_f = d_f / (d_f + d_b)
                e_b = d_b / (d_f + d_b)'''
                self.nodes.append((self.get_node_num(x, y, self.image.shape), 0, 0))

                '''if e_f > e_b:
                    self.graph[y, x] = 1.0
                else:
                    self.graph[y, x] = 0.0'''

        for (y, x), value in np.ndenumerate(self.graph):
            if y == self.graph.shape[0] - 1 or x == self.graph.shape[1] - 1:
                continue
            my_index = self.get_node_num(x, y, self.image.shape)

            neighbor_index = self.get_node_num(x+1, y, self.image.shape)
            g = 1 / (1 + np.sum(np.power(self.image[y, x] - self.image[y, x+1], 2)))
            self.edges.append((my_index, neighbor_index, g))

            neighbor_index = self.get_node_num(x, y+1, self.image.shape)
            g = 1 / (1 + np.sum(np.power(self.image[y, x] - self.image[y+1, x], 2)))
            self.edges.append((my_index, neighbor_index, g))

    def cut_graph(self):
        self.segment_overlay = np.zeros_like(self.segment_overlay)
        self.mask = np.zeros_like(self.image, dtype=bool)
        g = maxflow.Graph[float](len(self.nodes), len(self.edges))
        nodelist = g.add_nodes(len(self.nodes))

        for node in self.nodes:
            g.add_tedge(nodelist[node[0]], node[1], node[2])

        for edge in self.edges:
            g.add_edge(edge[0], edge[1], edge[2], edge[2])

        flow = g.maxflow()

        for index in range(len(self.nodes)):
            if g.get_segment(index) == 1:
                xy = self.get_xy(index, self.image.shape)
                self.segment_overlay[int(xy[1]),int(xy[0])] = (255, 0, 255)
                self.mask[int(xy[1]), int(xy[0])] = (True, True, True)

    def swap_overlay(self, overlay_num):
        self.current_overlay = overlay_num

    def save_image(self, filename):
        if self.mask is None:
            print ('Please segment the image before saving.')
            return

        to_save = np.zeros_like(self.image)

        np.copyto(to_save, self.image, where=self.mask)
        cv2.imwrite(str(filename), to_save)
        return to_save

    @staticmethod
    def get_node_num(x, y, array_shape):
        return y * array_shape[1] + x

    @staticmethod
    def get_xy(nodenum, array_shape):
        return (nodenum % array_shape[1]), (nodenum / array_shape[1])
    
class CutUI:

    def __init__(self, filename):
        self.graph_maker = GraphMaker((filename))
        self.display_image = np.array(self.graph_maker.image)
        self.window = "Graph Cut"
        self.mode = self.graph_maker.foreground
        self.started_click = False

    def run(self):
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.draw_line)

        while 1:
            display = cv2.addWeighted(self.display_image, 0.9, self.graph_maker.get_overlay(), 0.4, 0.1)
            cv2.imshow(self.window, display)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                break
            elif key == ord('c'):
                self.graph_maker.clear_seeds()
                
            elif key == ord('g'):
                out=self.graph_maker.create_graph()
                self.graph_maker.swap_overlay(self.graph_maker.segmented)
                cv2.destroyAllWindows()
                
                return(out)
            
            elif key == ord('t'):
                self.mode = 1 - self.mode
                self.graph_maker.swap_overlay(self.graph_maker.seeds)

        cv2.destroyAllWindows()

    def draw_line(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.started_click = True
            self.graph_maker.add_seed(x - 1, y - 1, self.mode)

        elif event == cv2.EVENT_LBUTTONUP:
            self.started_click = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.started_click:
                self.graph_maker.add_seed(x - 1, y - 1, self.mode)


#change filenames depending on which image you want to segment
filename='seg_samples/BloodImage_00029.jpg'
ui=CutUI(filename)
out=ui.run()
#out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
plt.imshow(out)
cv2.imwrite('seg_samples/GC_29.jpg', out)