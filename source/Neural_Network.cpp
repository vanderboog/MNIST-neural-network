#include <iostream>
#include <armadillo>
#include <string>
#include <memory>
#include "Neural_Network.h"

NN::NN() :
    setLearnSampleSize(100),
	setLearnReductionCycle(1000),
	setLearnReductionFactor(1),
	eta(0.1),
	regularization(0),
	halfRegularization(regularization / 2),
	iCountEpoch(0)
{}


void NN::setup(int numOfLayers, int *pLayerSize, std::string setPath)
{
	///////////////////////////////////////////////////////
	/// Creates layers and sets component sizes.
	/// Components are initialized ready for training
	//////////////////////////////////////////////////////
	setPathSave = setPath;

	/// Set hyperparameters to the correct values
	setNumOfLayers = numOfLayers;
	sizeLayer = std::unique_ptr<int[]>(new int[numOfLayers]);
	for (int iLayer = 0; iLayer < setNumOfLayers; iLayer++)
	{
		sizeLayer[iLayer] = pLayerSize[iLayer];
	}

	/// Create the layers and initialize parameters;
	pLayer = std::unique_ptr<cLayer[]>(new cLayer[setNumOfLayers]);
	pLayer[0].a.set_size(sizeLayer[0]); // Treat first layer different as it does not have b, w, nor kD
	for (int iLayer = 1; iLayer < setNumOfLayers; iLayer++)
	{
		// Init: set matrix and vector sizes
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

void NN::setHyperParameters(double setLearnSetSizeParam, double etaParam, double regularizationParam)
{
	setLearnSampleSize = setLearnSetSizeParam;
	eta = etaParam;
	regularization = regularizationParam;
	halfRegularization = regularization / 2;
	std::cout << "Hyper parameters settings:\n\t- Learning set size = " << setLearnSampleSize << "\n\t- Learning parameter (eta) = " << eta << "\n\t- Regularization parameter (Lambda) = " << regularization << "\n";
}

void NN::components()
{
	/// Outputs layers information
	std::cout << "Number of layers: \t" << setNumOfLayers << "\n";
	std::cout << "Number of neurons in layer 1: \t" << sizeLayer[0] << "\n";
	for (int iLayer = 1; iLayer < setNumOfLayers; iLayer++)
	{
		std::cout << "Number of neurons in layer " << iLayer + 1 << ": \t" << sizeLayer[iLayer] << "\n";
	}

	for (int iLayer = 1; iLayer < setNumOfLayers; iLayer++)
	{
		std::cout << "Weight matrix size (rows by cols) to layer " << iLayer + 1 << ": \t" << pLayer[iLayer].w.n_rows << " x " << pLayer[iLayer].w.n_cols << "\n";
	}
}

void NN::training(const arma::dmat &trainingSet, const arma::uvec &trainingLabels)
{
	///////////////////////////////////////////////////////
	/// Training the neural network by feeding it one epoch
	///////////////////////////////////////////////////////
	/// Init
	int numOfCol = trainingSet.n_cols;
	int numOfRow = trainingSet.n_rows;
	arma::uvec yVector(sizeLayer[setNumOfLayers - 1]);
	arma::uvec oneVector(sizeLayer[setNumOfLayers - 1], arma::fill::ones);
	arma::uvec sampleStack_i = arma::randperm(numOfCol);

	/// Reduce eta if -reduceLearnRate is used
	if (iCountEpoch % setLearnReductionCycle == 0 && iCountEpoch != 0)
	{
		reduceLearnRate(setLearnReductionFactor);
	}

	int numOfCyclesperEpoch = numOfCol / setLearnSampleSize; // Compute amount of cycles making up one epoch and only loop over complete cycles, omitting remaining samples
	/// Cycle through the epoch and apply learning after each cycle
	cost = arma::zeros(numOfCyclesperEpoch);
	for (int iCycle = 0; iCycle < numOfCyclesperEpoch; iCycle++)
	{
		int iSampleOffset = iCycle * setLearnSampleSize;

		/// Set dw and db to zero after each cycle
		for (int iLayer = 1; iLayer < setNumOfLayers; iLayer++)
		{
			pLayer[iLayer].db.zeros(pLayer[iLayer].db.n_rows, pLayer[iLayer].db.n_cols);
			pLayer[iLayer].dw.zeros(pLayer[iLayer].dw.n_rows, pLayer[iLayer].dw.n_cols);
		}

		for (int iSample = 0; iSample < setLearnSampleSize; iSample++)
		{
			/// Load the image and create label vector (yVector)
			pLayer[0].a = trainingSet.col(sampleStack_i(iSample + iSampleOffset));
			yVector = yVectorGenerator(trainingLabels(sampleStack_i(iSample + iSampleOffset)));

			/// Feed forward
			digit = feedForward(pLayer[0].a);

			/// Compute cost (-= is used instead of -1*)
			cost[iCycle] -= as_scalar(trans(yVector) * log(pLayer[setNumOfLayers - 1].a) + trans(oneVector - yVector) * log(oneVector - pLayer[setNumOfLayers - 1].a));
			/// Add regularization term:
			if (regularization != 0)  // Skip overhead computation in case of 0
			{
				for (int iLayer = 1; iLayer < setNumOfLayers; iLayer++)
				{
					cost[iCycle] += halfRegularization * accu(pLayer[iLayer].w % pLayer[iLayer].w);
				}
			}

			/// Back propagation
			/// Compute error terms: dC/dz
			pLayer[setNumOfLayers - 1].kD = pLayer[setNumOfLayers - 1].a - yVector;
			for (int iLayer = setNumOfLayers - 2; iLayer > 0; iLayer--)
			{
				pLayer[iLayer].kD = pLayer[iLayer + 1].w.t() * pLayer[iLayer + 1].kD % Dsigmoid(pLayer[iLayer].z);
			}
			/// Compute gradient descent of w and b (seperate loop for clarity)
			for (int iLayer = 1; iLayer < setNumOfLayers; iLayer++)
			{
				pLayer[iLayer].dw += arma::kron(pLayer[iLayer].kD, pLayer[iLayer - 1].a.t());
				pLayer[iLayer].db += pLayer[iLayer].kD;
			}
		}

		/// Apply gradient descent on w and b
		for (int iLayer = 1; iLayer < setNumOfLayers; iLayer++)
		{
			pLayer[iLayer].w -= eta * (pLayer[iLayer].dw + regularization * pLayer[iLayer].w) / setLearnSampleSize; // with regularization term
			pLayer[iLayer].b -= eta * pLayer[iLayer].db / setLearnSampleSize;
		}

		cost = cost / setLearnSampleSize;
	}
	iCountEpoch++;
}

arma::uvec NN::yVectorGenerator(const arma::uword &label)
{
	/// Generates a vector representation of the label: vector of zeros, with at the labelth index a 1
	arma::uvec y = arma::zeros<arma::uvec>(sizeLayer[setNumOfLayers - 1]);
	y(label) = 1;
	return y;
}

arma::dvec NN::sigmoid(arma::dvec &z)
{
	return 1 / (1 + exp(-z));
}

arma::dvec NN::Dsigmoid(arma::dvec &z)
{
	arma::dvec dS = sigmoid(z);
	return (dS % (1 - dS)); // %: Schur product, i.e. element-wise product
}

int NN::score(const arma::dmat &testSet, const arma::uvec &testLabels)
{
	////////////////////////////////////////////
	/// Score the network based on the test set
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
	std::cout << "Score: " << iCountCorrect << " / " << sizeSet << "\n";
	return iCountCorrect;
}

int NN::feedForward(const arma::dvec &imVector)
{
	/// Apply feedforward to determine and return the network answer
	pLayer[0].a = imVector;
	for (int iLayer = 1; iLayer < setNumOfLayers; iLayer++)
	{
		pLayer[iLayer].z = pLayer[iLayer].w * pLayer[iLayer - 1].a + pLayer[iLayer].b;
		pLayer[iLayer].a = sigmoid(pLayer[iLayer].z);
	}
	return pLayer[setNumOfLayers - 1].a.index_max();
}

void NN::setLearningReductionParameters(double setFactor, int setCycle)
{
	setLearnReductionFactor = setFactor;
	setLearnReductionCycle = setCycle;
	std::cout << "Learning rate reduction factor: " << setLearnReductionFactor << "\n";
	std::cout << "Learning rate reduction cycle: " << setLearnReductionCycle << "\n";
}

void NN::reduceLearnRate(double factor)
{
	eta = eta / factor;
	std::cout << "Eta reduced to:\t" << eta << "\n";
}

void NN::storeResults()
{
	/// Store essential parameters of the network: weights and biases
	for (int iLayer = 1; iLayer < setNumOfLayers; iLayer++)
	{
		pLayer[iLayer].w.save(setPathSave + "/w" + std::to_string(iLayer + 1));
		pLayer[iLayer].b.save(setPathSave + "/b" + std::to_string(iLayer + 1));
	}
}

void NN::loadResults(const std::string &setPathResult, int numOfLayers, int *layerSize)
{
	setPathSave = setPathResult;
	setNumOfLayers = numOfLayers;

	/// Load the actual stored data
	for (int iLayer = 1; iLayer < setNumOfLayers; iLayer++)
	{
		std::cout << "Loading file: " << (setPathResult + "/w" + std::to_string(iLayer + 1)) << "\n";
		pLayer[iLayer].w.load(setPathResult + "/w" + std::to_string(iLayer + 1));
		std::cout << "Loading file: " << (setPathResult + "/b" + std::to_string(iLayer + 1)) << "\n";
		pLayer[iLayer].b.load(setPathResult + "/b" + std::to_string(iLayer + 1));
	}

    components();
}
