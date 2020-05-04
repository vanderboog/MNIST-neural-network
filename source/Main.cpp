#include <armadillo>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <random>
#include <string>
#include "Neural_Network_Config.h"
#include "Neural_Network.h"
#include "ReadMNIST.h"
#include "Visualization.h"

std::string setPathSave(std::string const setPath)
{
    /// Make sure Result_Network directory exists
    if (!boost::filesystem::exists(setPath))
    {
        boost::filesystem::create_directory(setPath);
    }

    /// Set save path to a unique path of 'Results_##', found by incrementing from 1
    /// to 32. If the full range is used, the save path is set to 'Result_32'
    std::string setSavePath;
    for (int iFolder = 1; iFolder < 33; iFolder++)
    {
        setSavePath = setPath + "/Results_" + std::to_string(iFolder);
        if (!boost::filesystem::exists(setSavePath))
        {
            boost::filesystem::create_directory(setSavePath);
            break;
        }
    }

    std::cout << "Save path is set to: " << setSavePath << std::endl;
    return setSavePath;
}

void showUsage()
{
    std::cout << std::left << std::setw(92) << "Options available in this program:" << std::endl;
    std::cout << std::setw(2) << "" << std::setw(18) << "-train" << std::setw(72) << "Train a new neural network. This mode requires the training set and " << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "labels. See training options below for more details." << std::endl;
    std::cout << std::setw(2) << "" << std::setw(18) << "-test" << std::setw(72) << "Test a trained network. This mode requires a trained network stored in " << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "Results_Network and the test set. After '-test' refer to the folder " << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "containing the results by the trailing number in the folder name, e.g." << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "'-test 1' to test the network in 'Network_Results/Results_1'. See test " << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "options below for more details.\n"
              << std::endl;

    std::cout << std::left << std::setw(92) << "Training options: " << std::endl;
    std::cout << std::setw(2) << "" << std::setw(18) << "-layers" << std::setw(72) << "Set the total amount of layers and layer sizes used in the network," << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "including the input and output layer. After '-layers', the total number" << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "of layers is required. Thereafter, the layer size should be given in" << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "curly brackets, e.g. 'layers 3 {784,30,10}'." << std::endl;
    std::cout << std::setw(2) << "" << std::setw(18) << "-param" << std::setw(72) << "Set learning hyperparameters. Parameters which are to be set are: batch" << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "size before learning step, learning rate, and the regularization" << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "parameter, respectively. In case no regularization is to be used, the" << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "parameter is to be set to zero, e.g, '-param {1000,0.1,0}'" << std::endl;
    std::cout << std::setw(2) << "" << std::setw(18) << "-reduceLearning" << std::setw(72) << "Used to reduce the learning parameter by {factor x, per y epoch}," << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "e.g. -reduceLearning {2,20}.\n"
              << std::endl;

    std::cout << std::left << std::setw(92) << "Test options:" << std::endl;
    std::cout << std::setw(2) << "" << std::setw(18) << "-display" << std::setw(72) << "Opens a window to visualize the test images in a random sequence." << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "Visualization can be stopped by pressing <q>." << std::endl;
}

int main(int argc, const char **argv)
{
    /// Test if sufficient arguments are given
    if (argc < 2)
    {
        std::cerr << "No arguments are given. Use --help to show options.\nTerminating program." << std::endl;
        return 1;
    }
    else if (std::string(argv[1]) == "--help")
    {
        showUsage();
        return 0;
    }

    /// Initialize paths
    std::string const setPath = getCurrentDir(); // part of "readmnist.h"
    std::string const setPathTrainingImages = setPath + "/../Training_images/train-images.idx3-ubyte";
    std::string const setPathTrainingLabels = setPath + "/../Training_images/train-labels.idx1-ubyte";
    std::string const setPathTestImages = setPath + "/../Test_images/t10k-images.idx3-ubyte";
    std::string const setPathTestLabels = setPath + "/../Test_images/t10k-labels.idx1-ubyte";
    std::string const setPathResults = setPath + "/../Results_Network";
    std::string setPathConfig;

    // Initialize random engine for network initialization and random training/testing sequences
    std::default_random_engine generatorRandom{static_cast<long unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count())}; // Use high precision time to determine random seed

    if (std::string(argv[1]) == "-train")
    {
        setPathConfig = setPathSave(setPathResults);
    }
    else if(std::string(argv[1]) == "-test"){
        setPathConfig = setPathResults + "/Results_" + argv[2] + "/Input_parameters";
    }
    else {
        std::cerr << "The argument '" << argv[1] << "' is unknown to the program. Use --help to show viable options." << std::endl;
        return 2;
    }

    /// Setup network configuration and check if configuration succeeded
    NeuralNetworkConfig config(argc, argv, setPathConfig);
    if( config.bConfig == false){
        std::cerr << "Configuration is unsuccessful./nTerminating program..." << std::endl;
        return 1;
    }

    /// Use config to build the network
    NeuralNetwork network(config, generatorRandom);

    /// Interpret if program is used for training or testing
    if (std::string(argv[1]) == "-train")
    {
        std::cout << "Loading MNIST data...\n";

        // Reads images and returns a matrix(pxValue, numOfImages)
        std::cout << "Training data:" << std::endl;
        arma::dmat const trainingSet = readMnistImages(setPathTrainingImages);
        arma::uvec const trainingLabels = readMnistLabels(setPathTrainingLabels, trainingSet.n_cols);

        // Read test images to determine the score
        std::cout << "Test data:" << std::endl;
        arma::dmat const testSet = readMnistImages(setPathTestImages);
        arma::uvec const testLabels = readMnistLabels(setPathTestLabels, testSet.n_cols);

        /// Start training:
        int iCountScore = 0;
        int iEpocheCount = 0;
        while (iEpocheCount < 70)
        {
            // Perform a training cycle (one epoche)
            network.training(trainingSet, trainingLabels, generatorRandom);
            iEpocheCount += 1;

            std::cout << "Epoche counter: " << iEpocheCount << "\t\tAverage cost: " << arma::mean(network.cost) << std::endl;
            iCountScore = network.computePerformance(testSet, testLabels);

            /// Store results every epoche
            network.storeResults();
        }
    }
    else if (std::string(argv[1]) == "-test")
    {
        /// Load test files
        std::cout << "Loading MNIST data..."<< std::endl;
        std::cout << "Test data:" << std::endl;
        arma::dmat const testSet = readMnistImages(setPathTestImages);
        arma::uvec const testLabels = readMnistLabels(setPathTestLabels, testSet.n_cols);

        network.loadResults(setPathResults + "/Results_" + argv[2]);
        /// Compute and output the score
        network.computePerformance(testSet, testLabels);
        
        // Cycle through arguments given and apply settings
        for (int iArgc = 3; iArgc < argc; iArgc++)
        {
            if (std::string(argv[iArgc]) == "-display")
            {
                /// Display results in random order
                arma::arma_rng::set_seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
                arma::uvec sequence = arma::randperm(testSet.n_cols);

                int digit;
                std::string strDigit;
                int countDisplays = 0;
                arma::Mat<double> imArma;
                for (arma::uword iSequence : sequence)
                {
                    /// Run a sample through the network and obtain result
                    digit = -1;
                    digit = network.feedForward(testSet.col(iSequence));
                    strDigit = std::to_string(digit);

                    /// Reshape the image vector into a matrix and convert to openCV format
                    imArma = reshape(round(testSet.col(iSequence) * 256), 28, 28);
                    cv::Mat imDigit(28, 28, CV_64FC1, imArma.memptr());

                    /// Display the sample image with the networks answer
                    displayImage(imDigit, strDigit);
                    countDisplays++;

                    /// Give option to end the program
                    if (cv::waitKey(3000) == 'q')
                    {
                        break;
                    };
                }
            }
            else
            {
                std::cerr << "The argument '" << argv[iArgc] << "' is unknown to the program. Use --help to show viable options." << std::endl;
                return 2;
            }
        }
    }
    else
    {
        std::cerr << "The argument " << argv[1] << " is unknown to this program. Use --help to show viable options." << std::endl;
        return 2;
    }
    return 0;
}