struct CLayer
{
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

	//Smart pointers are used to ensure freeing of memory. The pointers are not always used and can therefore not be freed in a destructor
	std::unique_ptr<int[]> sizeLayer;
	std::unique_ptr<CLayer[]> pLayer;

public:
	arma::dvec cost;

	NeuralNetwork();

	void initializeLayers(int, int *, std::string);
	void setHyperParameters(double, double, double);
	void layerInfo();
	void training(const arma::dmat &, const arma::uvec &);
	arma::uvec yVectorGenerator(const arma::uword &);
	arma::dvec sigmoid(arma::dvec &);
	arma::dvec Dsigmoid(arma::dvec &);
	int computePerformance(const arma::dmat &, const arma::uvec &);
	int feedForward(const arma::dvec &);
	void setLearningReductionParameters(double, int);
	void reduceLearnRate(double);
	void storeResults();
	void loadResults(const std::string &, int, int *);
};
