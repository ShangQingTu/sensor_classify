The model_n_hidden_layer.py in the models folder represents a neural network with n hidden layers



Copy model.py completely from model_n_hidden_layer.py and replace it with model.py, meaning that the name of model.py under the sensor_calssify folder is still model.py, However, by changing the content to model_n_hidden_layer.py, the hidden layer of the neural network can be changed to N layer.



To change the number of nodes at each layer, add --dim_h 512 to the python train.py command to cut the number of neural network nodes at all layers in half (1024 by default).