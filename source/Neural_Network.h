class NeuralNetworkConfig;

class CLayer
{
	friend class NeuralNetwork;
	// Declaration of variables used in a each layer
	arma::dvec a;
	arma::dvec z;
	arma::dvec b;
	arma::dvec db;
	arma::dmat w;
	arma::dmat dw;
	arma::dvec kD;
};

class NeuralNetwork
{
	int numOfLayers_;
	int learnSetSize_;
	double learnRate_;
	double regularization_;
	double halfRegularization_;
	int learnReductionCycle_;
	double learnReductionFactor_;
	int iCountEpoch_;
	int digit_;
	std::string setSavePath_;
	std::vector<int> sizeLayer_;
	std::vector<CLayer> pLayer;

public:
	arma::dvec cost;

	NeuralNetwork( NeuralNetworkConfig &config, std::default_random_engine &generatorRandom);

	void layerInfo() const;
	void training(const arma::dmat &trainingSet, const arma::uvec &trainingLabels, std::default_random_engine & generatorRandom);
	arma::uvec yVectorGenerator(const arma::uword &label);
	arma::dvec sigmoid(arma::dvec &z);
	arma::dvec Dsigmoid(arma::dvec &z);
	int computePerformance(const arma::dmat &testSet, const arma::uvec &testLabels);
	int feedForward(const arma::dvec &imVector);
		void reduceLearnRate(const double &factor);
	void storeResults() const;
	void loadResults(const std::string &setSavePath);
};
