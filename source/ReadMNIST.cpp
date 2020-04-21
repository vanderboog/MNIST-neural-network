#include <iostream>
#include <armadillo>
#include <string>
#include "ReadMNIST.h"

#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

// Miscellaneous function
int ReverseInt(int iSample)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = iSample & 255;
    ch2 = (iSample >> 8) & 255;
    ch3 = (iSample >> 16) & 255;
    ch4 = (iSample >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

// Return a matrix containing the trainingset images. Format: (numOfImages, pxValue)
arma::Mat<double> ReadMNISTImages(std::string setFullPath)
{
    arma::umat arr;
    std::ifstream file(setFullPath, std::ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char *)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        std::cout << "Images in the set: " << number_of_images << "\n";
        std::cout << "Image size: " << n_rows << "*" << n_cols << "\n";
        arr.resize(number_of_images, n_rows * n_cols);

        for (int i = 0; i < number_of_images; ++i)
        {
            for (int r = 0; r < (n_rows * n_cols); ++r)
            {
                unsigned char temp = 0;
                file.read((char *)&temp, sizeof(temp));
                arr(i, r) = (double)temp;
            }
        }
    }
    return (arma::conv_to<arma::Mat<double> >::from(arr.t())/256);
}

// Return a column containing the labels per image
arma::uvec ReadMNISTLabels(std::string setPath, arma::uword numOfLabels)
{

    // arma::Col<long long int> sampleStack_i = arma::randi(10,arma::distr_param(0,10));
    // return sampleStack_i;

    arma::uvec vecLabel(numOfLabels);
    std::cout << "Number of labels: " << numOfLabels << "\n\n";

    std::ifstream file(setPath, std::ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int numOfLabels = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char *)&numOfLabels, sizeof(numOfLabels));
        numOfLabels = ReverseInt(numOfLabels);

        for (int iSample = 0; iSample < numOfLabels; ++iSample)
        {
            unsigned char temp = 0;
            file.read((char *)&temp, sizeof(temp));
            vecLabel(iSample) = (double)temp;
        }
    }
    return vecLabel;
}


std::string get_current_dir() {
   char buff[FILENAME_MAX]; //create string buffer to hold path
   GetCurrentDir( buff, FILENAME_MAX );
   std::string current_working_dir(buff);
   return current_working_dir;
}