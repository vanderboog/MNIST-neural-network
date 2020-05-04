#include <armadillo>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "Neural_Network_Config.h"

NeuralNetworkConfig::NeuralNetworkConfig(const int argc, const char **argv, std::string &setPathConfig) :
    // Default values
    setNumOfLayers_(1),
    setLearnSetSize_(100),
    setLearnRate_(0.1),
    setRegularization_(0),
    setLearnReductionCycle_(1000),
    setLearnReductionFactor_(1)
{
    /// Check if config is completed without issues
    bConfig = true;

    /// Interpret if program is used for training or testing
    if (std::string(argv[1]) == "-train")
    {
        /// Determine path to store results:
        setSavePath_ = setPathConfig;

        /// Store file containing input arguments:
        std::ofstream outputFile;
        outputFile.open(setSavePath_ + "/Input_parameters");
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
                setNumOfLayers_ = atoi(argv[iArgc + 1]);
                setSizeLayer_.resize(setNumOfLayers_);
                std::cout << "Layers found: \n";
                for (int iLayer = 0; iLayer < setNumOfLayers_; iLayer++)
                {
                    setSizeLayer_[iLayer] = atoi(argv[iArgc + 2 + iLayer]);
                    std::cout << setSizeLayer_[iLayer] << "\t";
                }
                std::cout << std::endl;
                iArgc += atoi(argv[iArgc + 1]) + 1;
            }
            else if (std::string(argv[iArgc]) == "-param")
            {
                /// Used to set hyperparameters directly related to learning { samplesize before learning, eta (learning rate), lambda (regulatization)}
                setLearnSetSize_ = atof(argv[iArgc + 1]);
                setLearnRate_ = atof(argv[iArgc + 2]);
                setRegularization_ = atof(argv[iArgc + 3]);
                iArgc += 3;
            }
            else if (std::string(argv[iArgc]) == "-reduceLearning")
            {
                /// Use to reduce learning rate at given intervals. Parameter order: { reduction factor, after # cycles }
                setLearnReductionFactor_ = atof(argv[iArgc + 1]);
                setLearnReductionCycle_ = atoi(argv[iArgc + 2]);
                iArgc += 2;
            }
            else
            {
                std::cerr << "The argument '" << argv[iArgc] << "' is unknown to the program. Use --help to show viable options." << std::endl;
                bConfig = false;
                break;
            }
        }
    }
    else if (std::string(argv[1]) == "-test")
    {
        std::ifstream inFile;
        inFile.open(setPathConfig);
        if (inFile.is_open())
        {
            /// Read parameters to determine set correct network size
            std::string arg;
            while (inFile >> arg)
            {
                if (arg == "-layers")
                {
                    inFile >> arg;
                    setNumOfLayers_ = stoi(arg);
                    setSizeLayer_.resize(setNumOfLayers_);
                    for (int iLayer = 0; iLayer < setNumOfLayers_; iLayer++)
                    {
                        inFile >> arg;
                        setSizeLayer_[iLayer] = stoi(arg);
                    }
                }
            }
            inFile.close();
        }
        else
        {
            std::cerr << "Unable to open a result file: " << setPathConfig << std::endl;
            bConfig = false;
        }
    }
}