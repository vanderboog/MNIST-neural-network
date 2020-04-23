#include <armadillo>
#include <iostream>
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
int reverseInt(int iSample)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = iSample & 255;
    ch2 = (iSample >> 8) & 255;
    ch3 = (iSample >> 16) & 255;
    ch4 = (iSample >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

// Return a matrix containing the trainingset images. Format: (numOfImages, pxValue)
arma::dmat readMnistImages(std::string setPath)
{
    arma::umat imSet;
    std::ifstream file(setPath, std::ios::binary);
    if (file.is_open())
    {
        int magicNumber = 0;
        int numOfImages = 0;
        int imRows = 0;
        int imCols = 0;
        file.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);
        file.read((char *)&numOfImages, sizeof(numOfImages));
        numOfImages = reverseInt(numOfImages);
        file.read((char *)&imRows, sizeof(imRows));
        imRows = reverseInt(imRows);
        file.read((char *)&imCols, sizeof(imCols));
        imCols = reverseInt(imCols);

        std::cout << "Images in the set: " << numOfImages << "\n";
        std::cout << "Image size: " << imRows << "*" << imCols << "\n";
        imSet.resize(numOfImages, imRows * imCols);

        for (int i = 0; i < numOfImages; ++i)
        {
            for (int r = 0; r < (imRows * imCols); ++r)
            {
                unsigned char input = 0;
                file.read((char *)&input, sizeof(input));
                imSet(i, r) = (double)input;
            }
        }
    }
    return (arma::conv_to<arma::dmat >::from(imSet.t())/256);
}

// Return a column containing the labels per image
arma::uvec readMnistLabels(std::string setPath, arma::uword numOfLabels)
{
    arma::uvec labelVector(numOfLabels);
    std::cout << "Number of labels: " << numOfLabels << "\n\n";

    std::ifstream file(setPath, std::ios::binary);
    if (file.is_open())
    {
        int magicNumber = 0;
        int numOfLabels = 0;
        file.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);
        file.read((char *)&numOfLabels, sizeof(numOfLabels));
        numOfLabels = reverseInt(numOfLabels);

        for (int iSample = 0; iSample < numOfLabels; ++iSample)
        {
            unsigned char input = 0;
            file.read((char *)&input, sizeof(input));
            labelVector(iSample) = (double)input;
        }
    }
    return labelVector;
}


std::string getCurrentDir() {
   char buff[FILENAME_MAX]; //create string buffer to hold path
   GetCurrentDir( buff, FILENAME_MAX );
   std::string currentWorkingDir(buff);
   return currentWorkingDir;
}