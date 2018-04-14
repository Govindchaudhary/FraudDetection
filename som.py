# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) #using normalisation 
X = sc.fit_transform(X)

# Training the SOM 
#using some other developer's implementation of SOM to make our life easy
from minisom import MiniSom
#here x and y are the dimension of our som(self-organizing map) the choice is pretty arbitrary bu it should not be very small
#input_len is the no. of features in our input set
#sigma is the radius of the different negihbourhood in the grid
#hyperparameter that decides by how much we update our weights during each iteration


som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)#our self organizing map
#randomly initializing the weights
som.random_weights_init(X)

""" Trains the SOM picking samples at random from data """
som.train_random(data = X, num_iteration = 100) #method to train the som on X,num_iteration is the no. of iterations to train our SOM

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone() #initializing the window to display the map

'''
putting all the winning neurons on the map
to do this we put info of mean into neuron distance of each winning node

mean into neuron distance(mid) of a specific winning neuron 
is the mean of the distances of the all the neurons 
around the winning node inside a neighbourhood that we define thanks to the sigma(radius of the neighbourhood)
basically higher is the mid means more the winning node is far away from his neighbours
also means that more the winning node is the outlier means the customer representing the node is the fraud
but we will not take into account the figures rather we use color
higher is the mid of the winning neuron more white will be the color of the wining neuron

but we will use color instead of figures
'''
pcolor(som.distance_map().T) #som.distance_map() func returns all the mean into neuron distances in one matrix and .T means we are taking the transpose
#in this case more white is the winning node more is the mid
colorbar()  #legend to indicate the what color represnts what value of mid

#adding the markers to identify which customers got approval and which dont
#bcos the customers who got approval and also are fraud are the more relevant for us

markers = ['o', 's']
colors = ['r', 'g'] #basically we are using red circle('o') to indicate customers who didn't got approval and green square to indicate that they do

'''
we are going to lop ove all the customers and for each customer
 we are going to get the winning node and depending on weather the customer 
 got approval or not we place our markers(cirlce or square)
 here i is the index of our customer and x is the vector represnting the customer
'''
for i, x in enumerate(X):
    w = som.winner(x) #get the wiining node of ist customer
    '''
      and now on this winning node we are going to place our marker based on weather he gets an approval or not
      now we are going to place our markers in the centre of winning node which is a square
      x-coordinate as x[0] and y as x[1] thse are the co-ordinates of lower left corner
      so we need to add .5 to these to get the centre
    '''

    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]], #we are going to use the circle if y[i]==0 means he didn't get approval and square if y[i]==1
         markeredgecolor = colors[y[i]], #color red if y[i]==0 otherwise green
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

'''
 Finding the frauds
we are going to use the dictionary that will contain all the mappings from winning nodes to the customer
basically this is the dictionary that contains the winning node co-ordinates as key
and array of customers assocaited to this winning node as value
'''
mappings = som.win_map(X)

'''getting the frauds(for this we have to get the co-ordinate of the outlying winning node) the most white ones
(mappings[(8,1)] gives us the list of all the customers assocaiting to this winning node
np.concataenate will combine the two list
axis is the axis alog which you want to conactenate the list either vertically
or horizontally,0 means vertically
'''

frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
#inverse transforming the frauds to get the original scale
frauds = sc.inverse_transform(frauds)