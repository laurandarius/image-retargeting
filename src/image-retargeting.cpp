//
//  main.cpp
//  image-retargeting
//
//  Created by Darius Lauran on 20/05/2019.
//  Copyright © 2019 Darius Lauran. All rights reserved.
//

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

Mat_<uchar> makeEnergyImage( Mat_<Vec3b> src )
{
  Mat_<uchar> src_gray;
  Mat_<uchar> grad;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  // Reduce noise
  GaussianBlur( src, src, Size( 3, 3 ), 0, 0, BORDER_DEFAULT );

  /// Convert it to gray
  cvtColor( src, src_gray, COLOR_BGR2GRAY );

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

  /*
  /// Create window
  char const* window_name = "Scharr Demo - Energy Image";
  namedWindow( window_name, WINDOW_AUTOSIZE );
  imshow( window_name, grad );
  */

  return grad;
}

float minUpperPixelIfIsInside( Mat_<float> src, int i, int j )
{
  if( i == 0 )
  {
    return 0.0;
  }
  if( j == 0 )
  {
    return min( src( i - 1, j ), src( i - 1, j + 1 ) );
  }
  if( j == src.cols - 1 )
  {
    return min( src( i - 1, j - 1 ), src( i - 1, j ) );
  }
  return min( min( src( i - 1, j - 1 ), src( i - 1, j ) ), src( i - 1, j + 1 ) );
}

Mat_<uchar> makeCumulativeEnergyMap( Mat_<uchar> src )
{
  float min_val = numeric_limits<float>::max(), max_val = 0.0;
  Mat_<float> cumulative_energy_map = Mat::zeros( Size( src.cols, src.rows ), CV_32FC1 );

  /// Loop top to bottom to cumulate energy
  for( int i = 0; i < src.rows; ++i )
  {
    for( int j = 0; j < src.cols; ++j )
    {
      cumulative_energy_map( i, j ) = (float)src( i, j ) + minUpperPixelIfIsInside( cumulative_energy_map, i, j );
      if( cumulative_energy_map( i, j ) > max_val )
      {
        max_val = cumulative_energy_map( i, j );
      }
      if( cumulative_energy_map( i, j ) < min_val )
      {
        min_val = cumulative_energy_map( i, j );
      }
    }
  }

  /// Normalise Mat_<float>
  cumulative_energy_map /= max_val;
  min_val /= max_val;
  max_val = 1.0;

  Mat_<uchar> cumulative_energy_map_gray;
  cumulative_energy_map.convertTo( cumulative_energy_map_gray, CV_8UC1, 255.0 / ( max_val - min_val ) );

  /*
  Mat_<Vec3b> cumulative_energy_map_bgr;

  /// Apply the colormap
  applyColorMap( cumulative_energy_map_gray, cumulative_energy_map_bgr, COLORMAP_JET );

  /// Create window
  char const* window_name = "Cumulative Energy Map";
  namedWindow( window_name, WINDOW_AUTOSIZE );
  imshow( window_name, cumulative_energy_map_bgr );
  */

  return cumulative_energy_map_gray;
}

bool isInside( Mat image, int i, int j )
{
  return i >= 0 && i < image.rows && j >= 0 && j < image.cols;
}

Mat_<Vec3b> removeSeam( Mat_<Vec3b> src, Mat_<uchar> cumulative_energy_map )
{
  Mat_<Vec3b> dest = src.clone();
  uchar min_val = 255;
  int min_energy_col = 0, col_to_be_removed;
  for( int i = 0; i < dest.cols; ++i )
  {
    if( cumulative_energy_map( cumulative_energy_map.rows - 1, i ) < min_val )
    {
      min_val = cumulative_energy_map( cumulative_energy_map.rows - 1, i );
      min_energy_col = i;
    }
  }

  col_to_be_removed = min_energy_col;

  for( int i = dest.rows - 1; i > 0; --i )
  {
    /// Choose upper min pixel on which the seam should be
    uchar upper_min_pixel = 255;
    for( int j = min_energy_col - 1; j <= min_energy_col + 1; ++j )
    {
      if( isInside( cumulative_energy_map, i - 1, j ) && cumulative_energy_map( i - 1, j ) < upper_min_pixel )
      {
        upper_min_pixel = cumulative_energy_map( i - 1, j );
        min_energy_col = j;
      }
    }

    /// Shift left or right so that in the end, one column will be removed
    if( min_energy_col < col_to_be_removed )
    {
      for( int j = min_energy_col; j < col_to_be_removed; ++j )
      {
        dest( i - 1, j ) = dest( i - 1, j + 1 );
      }
    }
    else
    {
      for( int j = min_energy_col; j > col_to_be_removed; --j )
      {
        dest( i - 1, j ) = dest( i - 1, j - 1 );
      }
    }
  }

  /// Remove column "col_to_be_removed" from destination image
  Mat before_cut, after_cut;
  dest( Range( 0, dest.rows ), Range( 0, col_to_be_removed ) ).copyTo( before_cut );
  dest( Range( 0, dest.rows ), Range( col_to_be_removed + 1, dest.cols ) ).copyTo( after_cut );
  hconcat( before_cut, after_cut, dest );

  return dest;
}

int main( int argc, char** argv )
{
  if( argc != 3 )
  {
    cerr << "Invalid number of arguments\n";
    return -1;
  }

  Mat_<Vec3b> image = imread( argv[1], IMREAD_COLOR );
  if( !image.data )
  {
    cerr << "Image not loaded\n";
    return -1;
  }
  int no_seams_to_remove = atoi( argv[2] );

  /// Create window
  char const* window_name_1 = "Load Image";
  namedWindow( window_name_1, WINDOW_AUTOSIZE );
  imshow( window_name_1, image );

  Mat_<Vec3b> resized = image.clone();
  for( int i = 0; i < no_seams_to_remove; ++i )
  {
    /// Energy Image
    Mat_<uchar> energy_image = makeEnergyImage( resized );

    /// Cumulative Energy Map
    Mat_<uchar> cumulative_energy_map = makeCumulativeEnergyMap( energy_image );

    resized = removeSeam( resized, cumulative_energy_map );
  }

  /// Create window
  char const* window_name_2 = "Retarget Image";
  namedWindow( window_name_2, WINDOW_AUTOSIZE );
  imshow( window_name_2, resized );

  waitKey();

  return 0;
}
