#include <armadillo>
#include <iostream>
#include <memory>
#include <string>
#include "Neural_Network.h"

NeuralNetwork::NeuralNetwork() :
    learnSetSize_(100),
	learnReductionCycle_(1000),
	learnReductionFactor_(1),
	learnRate_(0.1),
	regularization_(0),
	halfRegularization_(regularization_ / 2),
	iCountEpoch_(0)
{}


void NeuralNetwork::initializeLayers(int numOfLayers, int *pLayerSize, std::string setSavePath)
{
	///////////////////////////////////////////////////////
	/// Creates layers and sets component sizes.
	/// layerInfo are initialized ready for training
	//////////////////////////////////////////////////////
	setSavePath_ = setSavePath;
	numOfLayers_ = numOfLayers;
	sizeLayer = std::unique_ptr<int[]>(new int[numOfLayers_]);
	for (int iLayer = 0; iLayer < numOfLayers_; iLayer++)
	{
		sizeLayer[iLayer] = pLayerSize[iLayer];
	}

	/// Create the layers and initialize parameters;
	pLayer = std::unique_ptr<CLayer[]>(new CLayer[numOfLayers_]);
	pLayer[0].a.set_size(sizeLayer[0]); // Treat first layer different as it does not have b, w, nor kD
	for (int iLayer = 1; iLayer < numOfLayers_; iLayer++)
	{
		// Initialize: matrix and vector sizes
		pLayer[iLayer].a.set_size(sizeLayer[iLayer]);
		pLayer[iLayer].z.set_size(sizeLayer[iLayer]);
		pLayer[iLayer].b = arma::randn(sizeLayer[iLayer]);
		pLayer[iLayer].w.set_size(sizeLayer[iLayer], sizeLayer[iLayer - 1]);
		pLayer[iLayer].kD.set_size(sizeLayer[iLayer]);
		pLayer[iLayer].db = pLayer[iLayer].b;
		pLayer[iLayer].dw = pLayer[iLayer].w;

		/// Generate gaussian random generated values with standard deviation dependent on layer sizes.
		std::default_random_engine generator{static_cast<long unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count())}; // Use high precision time to determine random seed
		std::normal_distribution<double> distribution(0.0, sqrt((double)sizeLayer[iLayer - 1]));													// Generate random values of with stdev based on incoming layer
		for (arma::uword iRow = 0; iRow < sizeLayer[iLayer]; iRow++)
		{
			for (arma::uword iCol = 0; iCol < sizeLayer[iLayer - 1]; iCol++)
			{
				pLayer[iLayer].w(iRow, iCol) = distribution(generator);
			}
		}
	}
}

void NeuralNetwork::setHyperParameters(double learnSetSize, double learnRate, double regularization)
{
	learnSetSize_ = learnSetSize;
	learnRate_ = learnRate;
	regularization_ = regularization;
	halfRegularization_ = regularization_ / 2;
	std::cout << "Hyper parameters settings:\n\t- Learning set size = " << learnSetSize_ << "\n\t- Learning parameter (learnRate_) = " << learnRate_ << "\n\t- Regularization_ parameter (lambda) = " << regularization_ << "\n";
}

void NeuralNetwork::layerInfo()
{
	/// Outputs layers information
	std::cout << "Number of layers: \t" << numOfLayers_ << "\n";
	// std::cout << "Number of neurons in layer 1: \t" << sizeLayer[0] << "\n";
	for (int iLayer = 0; iLayer < numOfLayers_; iLayer++)
	{
		std::cout << "Number of neurons in layer " << iLayer + 1 << ": \t" << sizeLayer[iLayer] << "\n";
	}

	for (int iLayer = 1; iLayer < numOfLayers_; iLayer++)
	{
		std::cout << "Weight matrix size (rows by cols) to layer " << iLayer + 1 << ": \t" << pLayer[iLayer].w.n_rows << " x " << pLayer[iLayer].w.n_cols << "\n";
	}
}

void NeuralNetwork::training(const arma::dmat &trainingSet, const arma::uvec &trainingLabels)
{
	///////////////////////////////////////////////////////
	/// Training the neural network by feeding it one epoch
	///////////////////////////////////////////////////////
	/// Initialize
	int numOfCol = trainingSet.n_cols;
	int numOfRow = trainingSet.n_rows;
	arma::uvec yVector(sizeLayer[numOfLayers_ - 1]);
	arma::uvec oneVector(sizeLayer[numOfLayers_ - 1], arma::fill::ones);
	arma::uvec sampleStack_i = arma::randperm(numOfCol);

	/// Reduce learnRate_ if -reduceLearnRate is used
	if(iCountEpoch_ % learnReductionCycle_ == 0 && iCountEpoch_ != 0)
	{
		reduceLearnRate(learnReductionFactor_);
	}

	int numOfCyclesPerEpoch = numOfCol / learnSetSize_; // Compute amount of cycles making up one epoch and only loop over complete cycles, omitting remaining samples
	/// Cycle through the epoch and apply learning after each cycle
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
	arma::uvec y = arma::zeros<arma::uvec>(sizeLayer[numOfLayers_ - 1]);
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
	std::cout << "Performance: " << iCountCorrect << " / " << sizeSet << "\n";
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

void NeuralNetwork::setLearningReductionParameters(double learnReductionFactor, int learnReductionCycle)
{
	learnReductionFactor_ = learnReductionFactor;
	learnReductionCycle_ = learnReductionCycle;
	std::cout << "Learning rate reduction factor: " << learnReductionFactor_ << "\n";
	std::cout << "Learning rate reduction cycle: " << learnReductionCycle_ << "\n";
}

void NeuralNetwork::reduceLearnRate(double factor)
{
	learnRate_ = learnRate_ / factor;
	std::cout << "learnRate_ reduced to:\t" << learnRate_ << "\n";
}

void NeuralNetwork::storeResults()
{
	/// Store essential parameters of the network: weights and biases
	for (int iLayer = 1; iLayer < numOfLayers_; iLayer++)
	{
		pLayer[iLayer].w.save(setSavePath_ + "/w" + std::to_string(iLayer + 1));
		pLayer[iLayer].b.save(setSavePath_ + "/b" + std::to_string(iLayer + 1));
	}
}

void NeuralNetwork::loadResults(const std::string &setSavePath, int numOfLayers, int *layerSize)
{
	setSavePath_ = setSavePath;
	numOfLayers_ = numOfLayers;

	/// Load the actual stored data
	for (int iLayer = 1; iLayer < numOfLayers_; iLayer++)
	{
		std::cout << "Loading file: " << (setSavePath_ + "/w" + std::to_string(iLayer + 1)) << "\n";
		pLayer[iLayer].w.load(setSavePath_ + "/w" + std::to_string(iLayer + 1));
		std::cout << "Loading file: " << (setSavePath_ + "/b" + std::to_string(iLayer + 1)) << "\n";
		pLayer[iLayer].b.load(setSavePath_ + "/b" + std::to_string(iLayer + 1));
	}

    layerInfo();
}
