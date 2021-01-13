from tensorflow.keras.models import load_model
from visualizer import  Visualiser
import random

model = load_model("pneumonia_model_augmented5.h5")
img_path = "IM-0019-0001.jpeg"

# method that returns convolutional layers from model layers
def all_convolutional_layers_indices(model):
    model_layers = model.layers
    conv_layers_indices =[] 
    for layer_index ,layer in enumerate( model_layers):
        if str(layer).count("convolutional"):
            conv_layers_indices.append(layer_index)
            
    
    return conv_layers_indices


#method that returns ativation maps
def all_activation_maps(model):
    conv_layers_indices =all_convolutional_layers_indices(model)
    activation_maps_indices  = list(map(lambda x :  x+1  , conv_layers_indices))
        
    return activation_maps_indices

        
#convolutional layers indices
convolutional_layers = all_convolutional_layers_indices(model)  # [0,3,6,9]

#activation maps idnices 
activation_maps = all_activation_maps(model) # [1,4,7,9]


#instanciating

#instanciating a visualizer model with layers indices
#in this instance being the convolutional layers 
layers_indices =convolutional_layers
visualizer = Visualiser(img_path ,model  ,layers_indices )


#instanciating a visualizer model without layer indices
#adding layers indices after instanciating the object
layers_indices = convolutional_layers
visusalzer = Visualiser(img_path ,model , layers_indices)
visualizer.add_output_layers =  layers_indices


#either way the same result can be achieved



#remove layers indices
visualizer.remove_output_layers = convolutional_layers


#plot feature map / activation map of a specific layer
layers_indices = random.choice( [convolutional_layers ,activation_maps ] ) 
layer = 0
visualizer = Visualiser(img_path ,model , layers_indices)
visualizer.plot(layer)






            



