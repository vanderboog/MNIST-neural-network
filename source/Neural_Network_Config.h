class NeuralNetworkConfig {
	friend class NeuralNetwork;
	int setNumOfLayers_;
	std::vector<int> setSizeLayer_;
	int setLearnSetSize_;
	double setLearnRate_;
	double setRegularization_;
	int setLearnReductionCycle_;
	double setLearnReductionFactor_;
	int iCountEpoch_;
	

    public:
	std::string setSavePath_;
	bool bConfig;
    NeuralNetworkConfig( const int argc, const char **argv, std::string & setPathConfig);
};