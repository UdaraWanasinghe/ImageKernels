#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"


using namespace std;
using namespace cv;

void convolution_filter(Mat image, float kernel[3][3]);

float blur[3][3] = {{0.0625, 0.125, 0.0625},
                    {0.125,  0.25,  0.125},
                    {0.0625, 0.125, 0.0625}};

float bottom_sobel[3][3] = {{-1, -2, -1},
                            {0,  0,  0},
                            {1,  2,  1}};

float emboss[3][3] = {{-2, -1, 0},
                      {-1, 1,  1},
                      {0,  1,  2}};

float identity[3][3] = {{0, 0, 0},
                        {0, 1, 0},
                        {0, 0, 0}};

float left_sobel[3][3] = {{1, 0, -1},
                          {2, 0, -2},
                          {1, 0, -1}};

float outline[3][3] = {{-1, -1, -1},
                       {-1, 8,  -1},
                       {-1, -1, -1}};

float right_sobel[3][3] = {{-1, 0, 1},
                           {-2, 0, 2},
                           {-1, 0, 1}};

float sharpen[3][3] = {{0,  -1, 0},
                       {-1, 5,  -1},
                       {0,  -1, 0}};

float top_sobel[3][3] = {{1,  2,  1},
                         {0,  0,  0},
                         {-1, -2, -1}};

int main(int argc, char **argv) {
    if (argc < 2) {
        cout << "Image path is required" << endl;
        return 128;
    }
    if (argc < 3) {
        cout << "Filter type is required" << endl;
        cout << "0 - blur" << endl;
        cout << "1 - bottom sobel" << endl;
        cout << "2 - emboss" << endl;
        cout << "3 - identity" << endl;
        cout << "4 - left sobel" << endl;
        cout << "5 - outline" << endl;
        cout << "6 - right sobel" << endl;
        cout << "7 - sharpen" << endl;
        cout << "8 - top sobel" << endl;
        return 128;
    }
    string path = argv[1];
    int filterI = atoi(argv[2]);
    Mat image = imread(path, IMREAD_GRAYSCALE);
    switch (filterI) {
        case 0:
            convolution_filter(image, blur);
            break;
        case 1:
            convolution_filter(image, bottom_sobel);
            break;
        case 2:
            convolution_filter(image, emboss);
            break;
        case 3:
            convolution_filter(image, identity);
            break;
        case 4:
            convolution_filter(image, left_sobel);
            break;
        case 5:
            convolution_filter(image, outline);
            break;
        case 6:
            convolution_filter(image, right_sobel);
            break;
        case 7:
            convolution_filter(image, sharpen);
            break;
        case 8:
            convolution_filter(image, top_sobel);
            break;
    }
    imshow("Output", image);
    waitKey(0);

}

void convolution_filter(Mat image, float kernel[3][3]) {
    Mat clone = image.clone();
    for (int i = 1; i < image.rows - 1; i++) {
        for (int j = 1; j < image.rows - 1; j++) {
            float sum = 0;
            for (int ki = 0; ki < 3; ki++) {
                for (int kj = 0; kj < 3; kj++) {
                    int data = (int) clone.at<uchar>(i + ki - 1, j + kj - 1);
                    sum += (data * kernel[ki][kj]);
                }
            }
            image.at<uchar>(i, j) = (uchar) max(min(sum, 255.0F), 0.0F);
        }
    }
}