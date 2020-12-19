#include <ctime>
#include <iostream>

#include "Hough.h"
#include "circles.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define M_PI 3, 1415926535

using namespace cv;
using namespace std;

double compare(double val) {
  if (val < 0.0) return 0.0;
  if (val > 255.0) return 255.0;
  return val;
}

Mat GrayWorld(Mat& img) {
  int width = img.cols;
  int height = img.rows;
  cv::Mat result = cv::Mat(height, width, CV_8UC1);

  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      int B = img.at<Vec3b>(i, j)[0];
      int G = img.at<Vec3b>(i, j)[1];
      int R = img.at<Vec3b>(i, j)[2];

      double newValue = (R * 0.2126 + G * 0.7152 + B * 0.0722);
      result.at<uchar>(i, j) = newValue;
    }
  return result;
}

Mat MyMedianFilter(const Mat& src) {
  if (src.empty()) {
    cout << "Upload image" << endl;
    exit(-1);
  }
  Mat result = src.clone();
  int Rmatrix[9];
  int Gmatrix[9];
  int Bmatrix[9];
  for (int x = 1; x < src.rows - 1; x++) {
    for (int y = 1; y < src.cols - 1; y++) {
      Rmatrix[0] = src.at<Vec3b>(x - 1, y - 1)[0];
      Rmatrix[1] = src.at<Vec3b>(x, y - 1)[0];
      Rmatrix[2] = src.at<Vec3b>(x + 1, y - 1)[0];
      Rmatrix[3] = src.at<Vec3b>(x - 1, y)[0];
      Rmatrix[4] = src.at<Vec3b>(x, y)[0];
      Rmatrix[5] = src.at<Vec3b>(x + 1, y)[0];
      Rmatrix[6] = src.at<Vec3b>(x - 1, y + 1)[0];
      Rmatrix[7] = src.at<Vec3b>(x, y + 1)[0];
      Rmatrix[8] = src.at<Vec3b>(x + 1, y + 1)[0];

      Gmatrix[0] = src.at<Vec3b>(x - 1, y - 1)[1];
      Gmatrix[1] = src.at<Vec3b>(x, y - 1)[1];
      Gmatrix[2] = src.at<Vec3b>(x + 1, y - 1)[1];
      Gmatrix[3] = src.at<Vec3b>(x - 1, y)[1];
      Gmatrix[4] = src.at<Vec3b>(x, y)[1];
      Gmatrix[5] = src.at<Vec3b>(x + 1, y)[1];
      Gmatrix[6] = src.at<Vec3b>(x - 1, y + 1)[1];
      Gmatrix[7] = src.at<Vec3b>(x, y + 1)[1];
      Gmatrix[8] = src.at<Vec3b>(x + 1, y + 1)[1];

      Bmatrix[0] = src.at<Vec3b>(x - 1, y - 1)[2];
      Bmatrix[1] = src.at<Vec3b>(x, y - 1)[2];
      Bmatrix[2] = src.at<Vec3b>(x + 1, y - 1)[2];
      Bmatrix[3] = src.at<Vec3b>(x - 1, y)[2];
      Bmatrix[4] = src.at<Vec3b>(x, y)[2];
      Bmatrix[5] = src.at<Vec3b>(x + 1, y)[2];
      Bmatrix[6] = src.at<Vec3b>(x - 1, y + 1)[2];
      Bmatrix[7] = src.at<Vec3b>(x, y + 1)[2];
      Bmatrix[8] = src.at<Vec3b>(x + 1, y + 1)[2];

      sort(Rmatrix, Rmatrix + 9);
      sort(Gmatrix, Gmatrix + 9);
      sort(Bmatrix, Bmatrix + 9);

      result.at<Vec3b>(x, y)[0] = Rmatrix[4];
      result.at<Vec3b>(x, y)[1] = Gmatrix[4];
      result.at<Vec3b>(x, y)[2] = Bmatrix[4];
    }
  }
  return result;
}

Mat operatorSobel(const Mat& pic, Mat& angular_vec) {
  double x1[] = {1.0, 0, -1.0};
  double x2[] = {2.0, 0, -2.0};
  double x3[] = {1.0, 0, -1.0};

  vector<vector<double>> xFilter(3);

  xFilter[0].assign(x1, x1 + 3);
  xFilter[1].assign(x2, x2 + 3);
  xFilter[2].assign(x3, x3 + 3);

  double y1[] = {1.0, 2.0, 1.0};
  double y2[] = {0, 0, 0};
  double y3[] = {-1.0, -2.0, -1.0};

  vector<vector<double>> yFilter(3);

  yFilter[0].assign(y1, y1 + 3);
  yFilter[1].assign(y2, y2 + 3);
  yFilter[2].assign(y3, y3 + 3);

  int size = (int)xFilter.size() / 2;

  Mat result = Mat(pic.rows - 2 * size, pic.cols - 2 * size, CV_8UC1);
  angular_vec = Mat(pic.rows - 2 * size, pic.cols - 2 * size, CV_32FC1);

  for (int i = size; i < pic.rows - size; i++) {
    for (int j = size; j < pic.cols - size; j++) {
      double sumx = 0;
      double sumy = 0;

      for (int x = 0; x < xFilter.size(); x++)
        for (int y = 0; y < xFilter.size(); y++) {
          sumx += xFilter[x][y] *
                  (double)(pic.at<uchar>(i + x - size, j + y - size));
          sumy += yFilter[x][y] *
                  (double)(pic.at<uchar>(i + x - size, j + y - size));
        }
      double sumxsq = sumx * sumx;
      double sumysq = sumy * sumy;

      double sq2 = sqrt(sumxsq + sumysq);

      if (sq2 > 255) sq2 = 255;
      result.at<uchar>(i - size, j - size) = sq2;

      if (sumx == 0)
        angular_vec.at<float>(i - size, j - size) = 90;
      else
        angular_vec.at<float>(i - size, j - size) = atan(sumy / sumx);
    }
  }
  return result;
}
Mat nonMax(const Mat& pic, Mat& angular_vec) {
  Mat result = Mat(pic.rows - 2, pic.cols - 2, CV_8UC1);

  for (int i = 1; i < pic.rows - 1; i++) {
    for (int j = 1; j < pic.cols - 1; j++) {
      float Tangent = angular_vec.at<float>(i, j);

      result.at<uchar>(i - 1, j - 1) = pic.at<uchar>(i, j);
      // Horizontal Edge
      if (((-22.5 < Tangent) && (Tangent <= 22.5)) ||
          ((157.5 < Tangent) && (Tangent <= -157.5))) {
        if ((pic.at<uchar>(i, j) < pic.at<uchar>(i, j + 1)) ||
            (pic.at<uchar>(i, j) < pic.at<uchar>(i, j - 1)))
          result.at<uchar>(i - 1, j - 1) = 0;
      }
      // Vertical Edge
      if (((-112.5 < Tangent) && (Tangent <= -67.5)) ||
          ((67.5 < Tangent) && (Tangent <= 112.5))) {
        if ((pic.at<uchar>(i, j) < pic.at<uchar>(i + 1, j)) ||
            (pic.at<uchar>(i, j) < pic.at<uchar>(i - 1, j)))
          result.at<uchar>(i - 1, j - 1) = 0;
      }

      //-45
      if (((-67.5 < Tangent) && (Tangent <= -22.5)) ||
          ((112.5 < Tangent) && (Tangent <= 157.5))) {
        if ((pic.at<uchar>(i, j) < pic.at<uchar>(i - 1, j + 1)) ||
            (pic.at<uchar>(i, j) < pic.at<uchar>(i + 1, j - 1)))
          result.at<uchar>(i - 1, j - 1) = 0;
      }

      // 45
      if (((-157.5 < Tangent) && (Tangent <= -112.5)) ||
          ((22.5 < Tangent) && (Tangent <= 67.5))) {
        if ((pic.at<uchar>(i, j) < pic.at<uchar>(i + 1, j + 1)) ||
            (pic.at<uchar>(i, j) < pic.at<uchar>(i - 1, j - 1)))
          result.at<uchar>(i - 1, j - 1) = 0;
      }
    }
  }
  return result;
}

Mat doubleThresholdAndTrace(const Mat& pic, uint8_t min, uint8_t max) {
  min = compare(min);
  max = compare(max);
  Mat result = pic.clone();
  int width = pic.cols;
  int height = pic.rows;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      result.at<uchar>(i, j) = pic.at<uchar>(i, j);
      if (result.at<uchar>(i, j) > max)
        result.at<uchar>(i, j) = 255;
      else if (result.at<uchar>(i, j) < min)
        result.at<uchar>(i, j) = 0;
      else {
        bool high = false;
        bool between = false;
        for (int x = i - 1; x < i + 2; x++) {
          for (int y = j - 1; y < j + 2; y++) {
            if (x <= 0 || y <= 0 || height || y > width)
              continue;
            else {
              if (result.at<uchar>(x, y) > max) {
                result.at<uchar>(i, j) = 255;
                high = true;
                break;
              } else if (result.at<uchar>(x, y) <= max &&
                         result.at<uchar>(x, y) >= min)
                between = true;
            }
          }
          if (high) break;
        }
        if (!high && between)
          for (int x = i - 2; x < i + 3; x++) {
            for (int y = j - 1; y < j + 3; y++) {
              if (x < 0 || y < 0 || x > height || y > width)
                continue;
              else {
                if (result.at<uchar>(x, y) > max) {
                  result.at<uchar>(i, j) = 255;
                  high = true;
                  break;
                }
              }
            }
            if (high) break;
          }
        if (!high) result.at<uchar>(i, j) = 0;
      }
    }
  }
  return result;
}

Mat My_Canny(const Mat& pic, Mat& angular_vec) {
  Mat result = pic.clone();
  // 1st stage
  result = GrayWorld(result);

  // 2 stage
  result = MyMedianFilter(result);

  // 3 stage
  result = operatorSobel(result, angular_vec);

  // 4 stage
  result = nonMax(result, angular_vec);

  // 5 stage
  result = doubleThresholdAndTrace(result, 200, 210);

  return result;
}

struct SortCirclesDistance {
  bool operator()(const std::pair<std::pair<int, int>, int>& a,
                  const std::pair<std::pair<int, int>, int>& b) {
    int d = sqrt(pow(abs(a.first.first - b.first.first), 2) +
                 pow(abs(a.first.second - b.first.second), 2));
    if (d <= a.second + b.second) {
      return a.second > b.second;
    }
    return false;
  }
};

void HoughMethodLine(const cv::Mat src, const cv::Mat Canny) {
  int w = Canny.cols;
  int h = Canny.rows;

  // Transform
  myspace::Hough hough;
  hough.Transform(Canny.data, w, h);

  int threshold = w > h ? w / 4 : h / 4;

  while (1) {
    cv::Mat img_res = src.clone();

    // Search the accumulator
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> lines =
        hough.GetLines(threshold);

    // Draw the results
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>>::iterator
        it;
    for (it = lines.begin(); it != lines.end(); it++) {
      cv::line(img_res, cv::Point(it->first.first, it->first.second),
               cv::Point(it->second.first, it->second.second),
               cv::Scalar(0, 0, 255), 2, 8);
    }

    // Visualize all
    int aw, ah, maxa;
    aw = ah = maxa = 0;
    const unsigned int* accu = hough.GetAccu(&aw, &ah);

    for (int p = 0; p < (ah * aw); p++) {
      if ((int)accu[p] > maxa) maxa = accu[p];
    }
    double contrast = 1.0;
    double coef = 255.0 / (double)maxa * contrast;

    cv::Mat img_accu(ah, aw, CV_8UC3);
    for (int p = 0; p < (ah * aw); p++) {
      unsigned char c =
          (double)accu[p] * coef < 255.0 ? (double)accu[p] * coef : 255.0;
      img_accu.data[(p * 3) + 0] = 255;
      img_accu.data[(p * 3) + 1] = 255 - c;
      img_accu.data[(p * 3) + 2] = 255 - c;
    }

    const char* CW_IMG_ORIGINAL = "Result";
    const char* CW_ACCUMULATOR = "Accumulator";

    cv::imshow(CW_IMG_ORIGINAL, img_res);
    cv::imshow(CW_ACCUMULATOR, img_accu);

    char c = cv::waitKey(360000);
    if (c == 'p') threshold += 5;
    if (c == 'm') threshold -= 5;
    if (c == 27) break;
  }
}
void HoughMethodCircle(const Mat src, const Mat Canny) {
  int w = Canny.cols;
  int h = Canny.rows;

  const char* CW_IMG_ORIGINAL = "Result";
  const char* CW_ACCUMULATOR = "Accumulator";

  myspace::HoughCircle hough;

  vector<pair<pair<int, int>, int>> circles;
  Mat img_accu;
  for (int r = 19; r < h / 2; r = r + 1) {
    hough.Transform(Canny.data, w, h, r);

    cout << r << " / " << h / 2;

    int threshold = 0.95 * (2.0 * (double)r * M_PI);

    {
      hough.GetCircles(threshold, circles);

      int aw, ah, maxa;
      aw = ah = maxa = 0;
      const unsigned int* accu = hough.GetAccu(&aw, &ah);

      for (int p = 0; p < (ah * aw); p++) {
        if ((int)accu[p] > maxa) maxa = accu[p];
      }
      double contrast = 1.0;
      double coef = 255.0 / (double)maxa * contrast;
      img_accu = cv::Mat(ah, aw, CV_8UC3);
      for (int p = 0; p < (ah * aw); p++) {
        unsigned char c =
            (double)accu[p] * coef < 255.0 ? (double)accu[p] * coef : 255.0;
        img_accu.data[(p * 3) + 0] = 255;
        img_accu.data[(p * 3) + 1] = 255 - c;
        img_accu.data[(p * 3) + 2] = 255 - c;
      }
    }
  }

  sort(circles.begin(), circles.end(), SortCirclesDistance());
  int a, b, r;
  a = b = r = 0;
  vector<pair<std::pair<int, int>, int>> result;
  vector<pair<std::pair<int, int>, int>>::iterator it;
  for (it = circles.begin(); it != circles.end(); it++) {
    int d = sqrt(pow(abs(it->first.first - a), 2) +
                 pow(abs(it->first.second - b), 2));
    if (d > it->second + r) {
      result.push_back(*it);
      a = it->first.first;
      b = it->first.second;
      r = it->second;
    }
  }

  cv::Mat img_res = src.clone();
  for (it = result.begin(); it != result.end(); it++) {
    cout << it->first.first << ", " << it->first.second << std::endl;
    circle(img_res, Point(it->first.first, it->first.second), it->second,
           Scalar(0, 0, 255), 2, 8);
  }
  imshow(CW_IMG_ORIGINAL, img_res);
  imshow(CW_ACCUMULATOR, img_accu);
  waitKey(1);
}

int main() {
  // Task 1
  Mat angular_vec;
  Mat img1 = imread("D:/OI_lab3/Sea1.jpg", CV_LOAD_IMAGE_COLOR);
  imshow("Original image", img1);
  unsigned int my_start = clock();
  Mat img2 = My_Canny(img1, angular_vec);
  unsigned int my_end = clock();
  imshow("My Canny image", img2);
  unsigned int my_time = my_end - my_start;
  cout << "Time of work My_Canny: " << my_time << endl;
  Mat cv_result;
  unsigned int cv_start = clock();
  Canny(img1, cv_result, 80, 150);
  unsigned int cv_end = clock();
  imshow("CV Canny image", img2);
  unsigned int cv_time = cv_end - cv_start;
  cout << "Time of work CV_Canny: " << cv_time << endl;

  // Task 2
  Mat img3 = imread("D:/OI_lab3/circle.jpg", CV_LOAD_IMAGE_COLOR);
  HoughMethodLine(img1, img2);
  // HoughMethodCircle(img3, img2);
  waitKey(99999999);
  return 0;
}