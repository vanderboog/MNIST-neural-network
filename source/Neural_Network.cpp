#include <armadillo>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "Neural_Network.h"
#include "Neural_Network_Config.h"

NeuralNetwork::NeuralNetwork( NeuralNetworkConfig & config, std::default_random_engine & generatorRandom) :
    numOfLayers_(config.setNumOfLayers_),
	sizeLayer_(config.setSizeLayer_),
	learnSetSize_(config.setLearnSetSize_),
	learnReductionCycle_(config.setLearnReductionCycle_),
	learnReductionFactor_(config.setLearnReductionFactor_),
	learnRate_(config.setLearnRate_),
	regularization_(config.setRegularization_),
	halfRegularization_(regularization_ / 2),
	iCountEpoch_(0),
	setSavePath_(config.setSavePath_)
{
	///////////////////////////////////////////////////////
	/// Creates layers and sets component sizes.
	/// layerInfo are initialized ready for training
	//////////////////////////////////////////////////////

	/// Create the layers and initialize parameters;
	pLayer.resize(numOfLayers_);
	pLayer[0].a.set_size(sizeLayer_[0]); // Treat first layer different as it does not have b, w, nor kD
	for (int iLayer = 1; iLayer < numOfLayers_; iLayer++)
	{
		// Initialize: matrix and vector sizes
		pLayer[iLayer].a.set_size(sizeLayer_[iLayer]);
		pLayer[iLayer].z.set_size(sizeLayer_[iLayer]);
		pLayer[iLayer].b = arma::randn(sizeLayer_[iLayer]);
		pLayer[iLayer].w.set_size(sizeLayer_[iLayer], sizeLayer_[iLayer - 1]);
		pLayer[iLayer].kD.set_size(sizeLayer_[iLayer]);
		pLayer[iLayer].db = pLayer[iLayer].b;
		pLayer[iLayer].dw = pLayer[iLayer].w;

		/// Generate gaussian random generated values with standard deviation dependent on layer sizes.
		std::normal_distribution<double> distribution(0.0, sqrt((double)sizeLayer_[iLayer - 1])); // Generate random values of with stdev based on incoming layer
		for (arma::uword iRow = 0; iRow < sizeLayer_[iLayer]; iRow++)
		{
			for (arma::uword iCol = 0; iCol < sizeLayer_[iLayer - 1]; iCol++)
			{
				pLayer[iLayer].w(iRow, iCol) = distribution(generatorRandom);
			}
		}
	}

	layerInfo();
}

void NeuralNetwork::layerInfo() const
{
	/// Outputs layers information
	std::cout << "Number of layers: \t" << numOfLayers_ << std::endl;
	// std::cout << "Number of neurons in layer 1: \t" << sizeLayer_[0] << "\n";
	for (int iLayer = 0; iLayer < numOfLayers_; iLayer++)
	{
		std::cout << "Number of neurons in layer " << iLayer + 1 << ": \t" << sizeLayer_[iLayer] << std::endl;
	}

	for (int iLayer = 1; iLayer < numOfLayers_; iLayer++)
	{
		std::cout << "Weight matrix size (rows by cols) to layer " << iLayer + 1 << ": \t" << pLayer[iLayer].w.n_rows << " x " << pLayer[iLayer].w.n_cols << std::endl;
	}
	std::cout << std::endl;
}

void NeuralNetwork::training(const arma::dmat &trainingSet, const arma::uvec &trainingLabels, std::default_random_engine & generatorRandom )
{
	///////////////////////////////////////////////////////
	/// Training the neural network by feeding it one epoch
	///////////////////////////////////////////////////////
	/// Initialize
	int numOfCol = trainingSet.n_cols;
	int numOfRow = trainingSet.n_rows;
	arma::uvec yVector(sizeLayer_[numOfLayers_ - 1]);
	arma::uvec oneVector(sizeLayer_[numOfLayers_ - 1], arma::fill::ones);
	arma::uvec sampleStack_i = arma::randperm(numOfCol);

	/// Reduce learnRate_ if -reduceLearnRate is used
	if(iCountEpoch_ % learnReductionCycle_ == 0 && iCountEpoch_ != 0)
	{
		reduceLearnRate(learnReductionFactor_);
	}

	/// Cycle through the epoch and apply learning after each cycle
	int numOfCyclesPerEpoch = numOfCol / learnSetSize_; // Compute amount of cycles making up one epoch and only loop over complete cycles, omitting remaining samples
	cost = arma::zeros(numOfCyclesPerEpoch);
	for (int iCycle = 0; iCycle < numOfCyclesPerEpoch; iCycle++)
	{
		int iSampleOffset = iCycle * learnSetSize_;

		/// Set dw and db to zero before each cycle
		for (int iLayer = 1; iLayer < numOfLayers_; iLayer++)
		{
			pLayer[iLayer].db.zeros(pLayer[iLayer].db.n_rows, pLayer[iLayer].db.n_cols);
			pLayer[iLayer].dw.zeros(pLayer[iLayer].dw.n_rows, pLayer[iLayer].dw.n_cols);
		}

		for (int iSample = 0; iSample < learnSetSize_; iSample++)
		{
			/// Load the image and create label vector (yVector)
			pLayer[0].a = trainingSet.col(sampleStack_i(iSample + iSampleOffset));
			yVector = yVectorGenerator(trainingLabels(sampleStack_i(iSample + iSampleOffset)));

			/// Feed forward
			digit_ = feedForward(pLayer[0].a);

			/// Compute cost (-= is used instead of -1*)
			cost[iCycle] -= as_scalar(trans(yVector) * log(pLayer[numOfLayers_ - 1].a) + trans(oneVector - yVector) * log(oneVector - pLayer[numOfLayers_ - 1].a));
			/// Add regularization_ term:
			if (regularization_ != 0)  // Skip overhead computation in case of 0
			{
				for (int iLayer = 1; iLayer < numOfLayers_; iLayer++)
				{
					cost[iCycle] += halfRegularization_ * accu(pLayer[iLayer].w % pLayer[iLayer].w);  //Expensive
				}
			}

			/// Back propagation
			/// Compute error terms: dC/dz
			pLayer[numOfLayers_ - 1].kD = pLayer[numOfLayers_ - 1].a - yVector;
			for (int iLayer = numOfLayers_ - 2; iLayer > 0; iLayer--)
			{
				pLayer[iLayer].kD = pLayer[iLayer + 1].w.t() * pLayer[iLayer + 1].kD % Dsigmoid(pLayer[iLayer].z);
			}
			/// Compute gradient descent of w and b (seperate loop for clarity)
			for (int iLayer = 1; iLayer < numOfLayers_; iLayer++)
			{
				pLayer[iLayer].dw += arma::kron(pLayer[iLayer].kD, pLayer[iLayer - 1].a.t());
				pLayer[iLayer].db += pLayer[iLayer].kD;
			}
		}

		/// Apply gradient descent on w and b
		for (int iLayer = 1; iLayer < numOfLayers_; iLayer++)
		{
			pLayer[iLayer].w -= learnRate_ * (pLayer[iLayer].dw + regularization_ * pLayer[iLayer].w) / learnSetSize_; // with regularization_ term
			pLayer[iLayer].b -= learnRate_ * pLayer[iLayer].db / learnSetSize_;
		}
		cost = cost / learnSetSize_;
	}
	iCountEpoch_++;
}

arma::uvec NeuralNetwork::yVectorGenerator(const arma::uword &label)
{
	/// Generates a vector representation of the label: vector of zeros, with at the labelth index a 1
	arma::uvec y = arma::zeros<arma::uvec>(sizeLayer_[numOfLayers_ - 1]);
	y(label) = 1;
	return y;
}

arma::dvec NeuralNetwork::sigmoid(arma::dvec &z)
{
	return 1 / (1 + exp(-z));
}

arma::dvec NeuralNetwork::Dsigmoid(arma::dvec &z)
{
	arma::dvec dS = sigmoid(z);
	return (dS % (1 - dS)); // %: Schur product, i.e. element-wise product
}

int NeuralNetwork::computePerformance(const arma::dmat &testSet, const arma::uvec &testLabels)
{
	////////////////////////////////////////////
	/// Compute network performance based on the test set
	////////////////////////////////////////////

	int iCountCorrect = 0;
	int sizeSet = testSet.n_cols;
	for (int iSample = 0; iSample < sizeSet; iSample++)
	{
		// Load testimage & apply feedforward. Count the correct answers
		if (feedForward(testSet.col(iSample)) == testLabels(iSample))
		{
			iCountCorrect++;
		}
	}
	std::cout << "Performance: " << iCountCorrect << " / " << sizeSet << std::endl;
	return iCountCorrect;
}

int NeuralNetwork::feedForward(const arma::dvec &imVector)
{
	/// Apply feedforward to determine and return the network answer
	pLayer[0].a = imVector;
	for (int iLayer = 1; iLayer < numOfLayers_; iLayer++)
	{
		pLayer[iLayer].z = pLayer[iLayer].w * pLayer[iLayer - 1].a + pLayer[iLayer].b;
		pLayer[iLayer].a = sigmoid(pLayer[iLayer].z);
	}
	return pLayer[numOfLayers_ - 1].a.index_max();
}

void NeuralNetwork::reduceLearnRate(const double &factor)
{
	learnRate_ = learnRate_ / factor;
	std::cout << "Learn rate reduced to:\t" << learnRate_ << std::endl;
}

void NeuralNetwork::storeResults() const
{
	/// Store essential parameters of the network: weights and biases
	for (int iLayer = 1; iLayer < numOfLayers_; iLayer++)
	{
		pLayer[iLayer].w.save(setSavePath_ + "/w" + std::to_string(iLayer + 1));
		pLayer[iLayer].b.save(setSavePath_ + "/b" + std::to_string(iLayer + 1));
	}
}

void NeuralNetwork::loadResults(const std::string &setSavePath)
{
	setSavePath_ = setSavePath;
	
	/// Load the actual stored data
	std::cout << "Loading network files..." << std::endl;
	for (int iLayer = 1; iLayer < numOfLayers_; iLayer++)
	{
		std::cout << "Loading file: " << (setSavePath_ + "/w" + std::to_string(iLayer + 1)) << std::endl;
		pLayer[iLayer].w.load(setSavePath_ + "/w" + std::to_string(iLayer + 1));
		std::cout << "Loading file: " << (setSavePath_ + "/b" + std::to_string(iLayer + 1)) << std::endl;
		pLayer[iLayer].b.load(setSavePath_ + "/b" + std::to_string(iLayer + 1));
	}
}
