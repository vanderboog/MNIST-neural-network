// Header for reading the MNIST database in a matrix of (images,pixels)

arma::Mat<double> ReadMNISTImages( std::string);
arma::uvec ReadMNISTLabels( std::string, arma::uword );
std::string get_current_dir();