struct cLayer
{
	// Declaration of variables used in a each layer
	arma::Col<double> a;
	arma::Col<double> z;
	arma::Col<double> b;
	arma::Col<double> db;
	arma::dmat w;
	arma::dmat dw;
	arma::Col<double> kD;
};

class NN
{
	int setNumOfLayers;
	//Smart pointers are used to ensure freeing of memory. The pointers are not always used and can therefore not be freed in a destructor
	std::unique_ptr<int[]> sizeLayer;
	std::unique_ptr<cLayer[]> pLayer;

	int setLearnSampleSize;
	int setLearnReductionCycle;
	double setLearnReductionFactor;
	double eta;
	double regularization;
	double halfRegularization;
	int iCountEpoch;

	int digit;
	std::string setPathSave;

public:
	arma::dvec cost;

	NN();

	void setup(int, int *, std::string);
	void setHyperParameters(double, double, double);
	void components();
	void training(const arma::dmat &, const arma::uvec &);
	arma::uvec yVectorGenerator(const arma::uword &);
	arma::dvec sigmoid(arma::dvec &);
	arma::dvec Dsigmoid(arma::dvec &);
	int score(const arma::dmat &, const arma::uvec &);
	int feedForward(const arma::dvec &);
	void setLearningReductionParameters(double, int);
	void reduceLearnRate(double);
	void storeResults();
	void loadResults(const std::string &, int, int *);
};
