#include <iostream>
#include <armadillo>
#include <string>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <boost/filesystem.hpp>
#include <random>
#include <memory>
#include <iomanip>
#include "ReadMNIST.h"
#include "Visualization.h"
#include "Neural_Network.h"

std::string setSavePath(std::string setPath)
{
    /// Make sure Result_Network directory exists
    if (!boost::filesystem::exists(setPath))
    {
        boost::filesystem::create_directory(setPath);
    }

    /// Set save path to a unique path of 'Results_##', found by incrementing from 1 
    /// to 32. If the full range is used, the save path is set to 'Result_32'
    std::string setPathSave;
    for (int iFolder = 1; iFolder < 33; iFolder++)
    {
        setPathSave = setPath + "/Results_" + std::to_string(iFolder);
        if (!boost::filesystem::exists(setPathSave))
        {
            boost::filesystem::create_directory(setPathSave);
            break;
        }
    }

    std::cout << "Save path is set to: " << setPathSave << "\n";
    return setPathSave;
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
    std::cout << std::setw(2)  << "" << std::setw(18) << "-layers" << std::setw(72) << "Set the total amount of layers and layer sizes used in the network," << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "including the input and output layer. After '-layers', the total number" << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "of layers is required. Thereafter, the layer size should be given in" << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "curly brackets, e.g. 'layers 3 {784,30,10}'." << std::endl;
    std::cout << std::setw(2)  << "" << std::setw(18) << "-param" << std::setw(72) << "Set learning hyperparameters. Parameters which are to be set are: batch" << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "size before learning step, learning rate, and the regularization" << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "parameter, respectively. In case no regularization is to be used, the" << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "parameter is to be set to zero, e.g, '-param {1000,0.1,0}'" << std::endl;
    std::cout << std::setw(2)  << "" << std::setw(18) << "-reduceLearning" << std::setw(72) << "Used to reduce the learning parameter by {factor x, per y epoch}," << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "e.g. -reduceLearning {2,20}.\n"
              << std::endl;

    std::cout << std::left << std::setw(92) << "Test options:" << std::endl;
    std::cout << std::setw(2)  << "" << std::setw(18) << "-display" << std::setw(72) << "Opens a window to visualize the test images in a random sequence." << std::endl;
    std::cout << std::setw(20) << "" << std::setw(72) << "Visualization can be stopped by pressing <q>." << std::endl;
}

int main(int argc, const char **argv)
{
    /// Test if sufficient arguments are given
    if (argc < 2)
    {
        std::cout << "No arguments are given. Use --help to show options.\nTerminating program.\n";
        return 1;
    }

    /// Init paths
    std::string const setPath = get_current_dir(); // part of "ReadMNIST.h"
    std::string const setPathTrainingImages = setPath + "/../Training_images/train-images.idx3-ubyte";
    std::string const setPathTrainingLabels = setPath + "/../Training_images/train-labels.idx1-ubyte";
    std::string const setPathTestImages = setPath + "/../Test_images/t10k-images.idx3-ubyte";
    std::string const setPathTestLabels = setPath + "/../Test_images/t10k-labels.idx1-ubyte";
    std::string const setPathResults = setPath + "/../Results_Network";

    NN network;

    /// Interpret if program is used for training or testing
    if (std::string(argv[1]) == "-train")
    {
        /// Determine path to store results:
        std::string setPathStoreResult = setSavePath(setPathResults);

        /// Store file containing input arguments:
        std::ofstream outputFile;
        outputFile.open(setPathStoreResult + "/Input_parameters");
        for (int iArgv = 2; iArgv < argc + 1; iArgv++)
        {
            outputFile << argv[iArgv] << "\t";
        }
        outputFile.close();

        // Cycle through arguments given and apply settings to the neural network
        for (int iArgc = 2; iArgc < argc; iArgc++)
        {
            if (std::string(argv[iArgc]) == "-layers")
            {
                /// Used to set the layers of the neural network.
                /// The first trailing argument should be the amount of layers. Subsequent the layer sizes are to be given in seperate arguments,
                /// starting from the input layer, up to the output layer. E.g. '-layers 3 {784,30,10}'
                int *pLayers = new int[atoi(argv[iArgc + 1])];
                std::cout << "Layers found: \n";
                for (int iLayer = 0; iLayer < atoi(argv[iArgc + 1]); iLayer++)
                {
                    pLayers[iLayer] = atoi(argv[iArgc + 2 + iLayer]);
                    std::cout << pLayers[iLayer] << "\t";
                }
                std::cout << "\n";
                network.setup(atoi(argv[iArgc + 1]), pLayers, setPathStoreResult);
                delete[] pLayers;
                network.components();
                iArgc += atoi(argv[iArgc + 1]) + 1;
            }
            else if (std::string(argv[iArgc]) == "-param")
            {
                /// Used to set hyperparameters directly related to learning { samplesize before learning, eta (learning rate), lambda (regulatization)}
                network.setHyperParameters(atof(argv[iArgc + 1]), atof(argv[iArgc + 2]), atof(argv[iArgc + 3]));
                iArgc += 3;
            }
            else if (std::string(argv[iArgc]) == "-reduceLearning")
            {
                /// Use to reduce learning rate at given intervals. Parameter order: { reduction factor, after # cycles }
                network.setLearningReductionParameters(atof(argv[iArgc + 1]), atoi(argv[iArgc + 2]));
                iArgc += 2;
            }
            else
            {
                std::cout << "The argument '" << argv[iArgc] << "' is unknown to the program. Use --help to show viable options." << std::endl;
                return 0;
            }
        }

        /// Load data for training:
        std::cout << "Loading data...\n";
        // Reads images and returns a matrix(pxValue, numOfImages)
        arma::dmat const trainingSet = ReadMNISTImages(setPathTrainingImages);
        arma::uvec const trainingLabels = ReadMNISTLabels(setPathTrainingLabels, trainingSet.n_cols);

        // Read test images to determine the score
        arma::dmat const testSet = ReadMNISTImages(setPathTestImages);
        arma::uvec const testLabels = ReadMNISTLabels(setPathTestLabels, testSet.n_cols);

        /// Start training:
        int iCountScore = 0;
        long int iEpocheCount = 0;
        while (iEpocheCount < 70)
        {
            // Perform a training cycle (one epoche)
            network.training(trainingSet, trainingLabels);
            iEpocheCount += 1;

            std::cout << "Epoche counter: " << iEpocheCount << "\t\tAverage cost: " << arma::mean(network.cost) << "\n";
            iCountScore = network.score(testSet, testLabels);

            /// Store results every epoche
            network.storeResults();
        }
    }

    else if (std::string(argv[1]) == "-test")
    {
        /// Load test files
        std::cout << "Loading data...\n";
        arma::dmat const testSet = ReadMNISTImages(setPathTestImages);
        arma::uvec const testLabels = ReadMNISTLabels(setPathTestLabels, testSet.n_cols);

        /// Read parameters from parameter file
        std::ifstream inFile;
        std::string setPathToLoad = setPathResults + "/Results_" + argv[2] + "/Input_parameters";

        inFile.open(setPathToLoad);
        if (inFile.is_open())
        {
            /// Read parameters to determine set correct network size
            int numOfLayers;
            int *pLayer;
            std::string arg;
            while (inFile >> arg)
            {
                if (arg == "-layers")
                {
                    inFile >> arg;
                    numOfLayers = stoi(arg);
                    pLayer = new int[numOfLayers];
                    for (int iLayer = 0; iLayer < numOfLayers; iLayer++)
                    {
                        inFile >> arg;
                        pLayer[iLayer] = stoi(arg);
                    }

                    /// Initialize weights and biases sizes and load results
                    network.setup(numOfLayers, pLayer, setPathResults + "/Results_" + argv[2]);
                    network.loadResults(setPathResults + "/Results_" + argv[2], numOfLayers, pLayer);
                }
            }
            /// Compute and output the score
            network.score(testSet, testLabels);
            inFile.close();
            delete[] pLayer;
        }
        else
        {
            std::cout << "Unable to open a result file: " << setPathToLoad << "\n";
            return 2;
        }

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
                std::cout << "The argument '" << argv[iArgc] << "' is unknown to the program. Use --help to show viable options." << std::endl;
                return 0;
            }
        }
    }
    else if (std::string(argv[1]) == "--help")
    {
        showUsage();
    }
    else
    {
        std::cerr << "The argument " << argv[1] << " is unknown to this program. Use --help to show viable options." << std::endl;
    }
    return 0;
}