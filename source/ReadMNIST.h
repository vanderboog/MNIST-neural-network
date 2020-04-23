// Header for reading the MNIST database in a matrix of (images,pixels)

arma::dmat readMnistImages( std::string);
arma::uvec readMnistLabels( std::string, arma::uword );
std::string getCurrentDir();