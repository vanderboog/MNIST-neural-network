# mnist-neural-network
Neural network implementation in C++ for handwritten digit recognition using the MNIST database. Use the program to train, test and visualize results of a neural network trained using mnist data. The program allows to set the layer depth and sizes. The network implements a cross-entropy cost function with regularization. For more details, see the option available in this program.

Usage (example):
train example:  ./Neural_network -train 3 {784,30,10} -param {10,0.01,0.01} -reduceLearning {2,20}
test example:   ./Neural_network -test 1 -display

Packages used:
- Armadillo (linear algebra)
- OpenCV (visualization)

Options available in this program:                                                          
  -train            Train a new neural network. This mode requires the training set and     
                    labels. See training options below for more details.                    
  -test             Test a trained network. This mode requires a trained network stored in  
                    Results_Network and the test set. After '-test' refer to the folder     
                    containing the results by the trailing number in the folder name, e.g.  
                    '-test 1' to test the network in 'Network_Results/Results_1'. See test  
                    options below for more details.
                                        
Training options:                                                                           
  -layers           Set the total amount of layers and layer sizes used in the network,     
                    including the input and output layer. After '-layers', the total number 
                    of layers is required. Thereafter, the layer size should be given in    
                    curly brackets, e.g. 'layers 3 {784,30,10}'.                            
  -param            Set learning hyperparameters. Parameters which are to be set are: batch 
                    size before learning step, learning rate, and the regularization        
                    parameter, respectively. In case no regularization is to be used, the   
                    parameter is to be set to zero, e.g, '-param {1000,0.1,0}'              
  -reduceLearning   Used to reduce the learning parameter by {factor x, per y epoch},       
                    e.g. -reduceLearning {2,20}.
                                           
Test options:                                                                               
  -display          Opens a window to visualize the test images in a random sequence.       
                    Visualization can be stopped by pressing <q>.     


