//
//  main.cpp
//  image-retargeting
//
//  Created by Darius Lauran on 20/05/2019.
//  Copyright © 2019 Darius Lauran. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

Mat makeEnergyImage( Mat src )
{
  Mat src_gray;
  Mat grad;
  char const* window_name = "Scharr Demo - Energy Image";
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  // Reduce noise
  GaussianBlur( src, src, Size( 3, 3 ), 0, 0, BORDER_DEFAULT );

  /// Convert it to gray
  cvtColor( src, src_gray, COLOR_BGR2GRAY );

  /// Create window
  namedWindow( window_name, WINDOW_AUTOSIZE );

  /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  /// Gradient X
  Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );

  /// Total Gradient (approximate)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

  imshow( window_name, grad );

  return grad;
}

uchar minPrevPixelIfIsInside( Mat src, int i, int j )
{
  if( i == 0 )
  {
    return src.at<uchar>( i, j );
  }
  if( j == 0 )
  {
    return min( src.at<uchar>( i - 1, j ), src.at<uchar>( i - 1, j + 1 ) );
  }
  if( j == src.cols - 1 )
  {
    return min( src.at<uchar>( i - 1, j - 1 ), src.at<uchar>( i - 1, j ) );
  }

  return min( min(src.at<uchar>( i - 1, j - 1 ), src.at<uchar>( i - 1, j )), src.at<uchar>( i - 1, j + 1 ) );
}

Mat makeCumulativeEnergyMap( Mat src )
{
  Mat src_bgr;
  char const* window_name = "Cumulative Energy Map";

  /// Create window
  namedWindow( window_name, WINDOW_AUTOSIZE );

  /// Loop top to bottom
  for( int i = 0; i < src.rows; ++i )
  {
    for( int j = 0; j < src.cols; ++j )
    {
      src.at<uchar>( i, j ) += minPrevPixelIfIsInside( src, i, j );
      if( src.at<uchar>( i, j ) >= 255 )
      {
        src.at<uchar>( i, j ) = 255;
      }
    }
  }

  /// Apply the colormap
  applyColorMap( src, src_bgr, COLORMAP_JET );

  imshow( window_name, src_bgr );

  return src;
}

int main( int argc, char** argv )
{
  if( argc != 2 )
  {
    cerr << "Invalid number of arguments\n";
    return -1;
  }

  Mat image = imread( argv[1], IMREAD_COLOR );
  if( !image.data )
  {
    cerr << "Image not loaded\n";
    return -1;
  }
  char const* window_name_1 = "Load Image";
  namedWindow( window_name_1, WINDOW_AUTOSIZE );
  imshow( window_name_1, image );

  /// Enery Image
  image = makeEnergyImage( image );

  /// Cumulative Energy Map
  image = makeCumulativeEnergyMap( image );

  waitKey();

  return 0;
}
