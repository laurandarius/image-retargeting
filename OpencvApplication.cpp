//
//  image-retargeting.cpp
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

enum SeamDirection { VERTICAL, HORIZONTAL };

Mat_<double> makeEnergyImage( Mat_<Vec3b> src ) {
  Mat_<uchar> src_gray;
  Mat_<Vec3b> src_blur;
  Mat grad;
  Mat_<double> dest;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  // Reduce noise
  GaussianBlur( src, src_blur, Size( 3, 3 ), 0, 0, BORDER_DEFAULT );

  /// Convert it to gray
  cvtColor( src_blur, src_gray, COLOR_BGR2GRAY );

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

  // convert the default values to double precision
  grad.convertTo(dest, CV_64F, 1.0 / 255.0);

  return dest;
}

double minUpperPixelIfIsInside( Mat_<double> img, int i, int j ) {
  if( i == 0 ) {
    return 0.0;
  }
  double pix1 = img( i - 1, max( j - 1, 0 ) );
  double pix2 = img( i - 1, j );
  double pix3 = img( i - 1, min( j + 1, img.cols - 1 ) );

  return min( min( pix1, pix2 ), pix3 );
}

Mat_<double> makeCumulativeEnergyMap( Mat_<double> src ) {
  double min_val = numeric_limits<double>::max(), max_val = 0.0;
  Mat_<double> cumulative_energy_map = Mat::zeros( Size( src.cols, src.rows ), CV_64F );

  /// Loop top to bottom to cumulate energy
  for( int i = 0; i < src.rows; ++i ) {
    for( int j = 0; j < src.cols; ++j ) {
      cumulative_energy_map( i, j ) = src( i, j ) + minUpperPixelIfIsInside( cumulative_energy_map, i, j );
      if( cumulative_energy_map( i, j ) > max_val ) {
        max_val = cumulative_energy_map( i, j );
      }
      if( cumulative_energy_map( i, j ) < min_val ) {
        min_val = cumulative_energy_map( i, j );
      }
    }
  }

  return cumulative_energy_map;
}

bool isInside( Mat image, int i, int j ) {
  return i >= 0 && i < image.rows && j >= 0 && j < image.cols;
}

void removeSeam( Mat_<Vec3b> &img, Mat_<double> cumulative_energy_map, SeamDirection seam_direction ) {
  double min_val = numeric_limits<double>::max();
  int min_energy_line = 0, to_be_removed;
  if ( seam_direction == VERTICAL ) {
    for( int i = 0; i < img.cols; ++i ) {
      if( cumulative_energy_map( cumulative_energy_map.rows - 1, i ) < min_val ) {
        min_val = cumulative_energy_map( cumulative_energy_map.rows - 1, i );
        min_energy_line = i;
      }
    }
    to_be_removed = min_energy_line;
    for( int i = img.rows - 1; i > 0; --i ) {
      /// Choose upper minpixel on which the seam should be
      double next_min_pixel = numeric_limits<double>::max();
      for( int j = min_energy_line - 1; j <= min_energy_line + 1; ++j ) {
        if( isInside( cumulative_energy_map, i - 1, j ) && cumulative_energy_map( i - 1, j ) < next_min_pixel ) {
          next_min_pixel = cumulative_energy_map( i - 1, j );
          min_energy_line = j;
        }
      }

      /// Shift left or right so that in the end, one column will be removed
      if( min_energy_line < to_be_removed ) {
        for( int j = min_energy_line; j < to_be_removed; ++j ) {
          img( i - 1, j ) = img( i - 1, j + 1 );
        }
      }
      else {
        for( int j = min_energy_line; j > to_be_removed; --j ) {
          img( i - 1, j ) = img( i - 1, j - 1 );
        }
      }
    }

    /// Remove column "to_be_removed" from image
    Mat before_cut, after_cut;
    img( Range( 0, img.rows ), Range( 0, to_be_removed ) ).copyTo( before_cut );
    img( Range( 0, img.rows ), Range( to_be_removed + 1, img.cols ) ).copyTo( after_cut );
    if( !before_cut.empty() && !after_cut.empty() ) {
      hconcat( before_cut, after_cut, img );
    }
    else {
      if( before_cut.empty() ) {
        img = after_cut;
      } else if( after_cut.empty() ) {
        img = before_cut;
      }
    }
  } else if ( seam_direction == HORIZONTAL ) {
    for( int i = 0; i < img.rows; ++i ) {
      if( cumulative_energy_map( i, cumulative_energy_map.cols - 1 ) < min_val ) {
        min_val = cumulative_energy_map( i, cumulative_energy_map.cols - 1 );
        min_energy_line = i;
      }
    }
    to_be_removed = min_energy_line;
    for( int i = img.cols - 1; i > 0; --i ) {
      /// Choose upper minpixel on which the seam should be
      double next_min_pixel = numeric_limits<double>::max();
      for( int j = min_energy_line - 1; j <= min_energy_line + 1; ++j ) {
        if( isInside( cumulative_energy_map, j - 1, i ) && cumulative_energy_map( j - 1, i ) < next_min_pixel ) {
          next_min_pixel = cumulative_energy_map( j - 1, i );
          min_energy_line = j;
        }
      }

      /// Shift left or right so that in the end, one column will be removed
      if( min_energy_line < to_be_removed ) {
        for( int j = min_energy_line; j < to_be_removed; ++j ) {
          img( j - 1, i ) = img( j - 1, i + 1 );
        }
      }
      else {
        for( int j = min_energy_line; j > to_be_removed; --j ) {
          img( j - 1, i ) = img( j - 1, i - 1 );
        }
      }
    }

    /// Remove row "to_be_removed" from image
    Mat before_cut, after_cut;
    img( Range( 0, to_be_removed ), Range( 0, img.cols ) ).copyTo( before_cut );
    img( Range( to_be_removed + 1, img.rows ), Range( 0, img.cols ) ).copyTo( after_cut );
    if( !before_cut.empty() && !after_cut.empty() ) {
      vconcat( before_cut, after_cut, img );
    }
    else {
      if( before_cut.empty() ) {
        img = after_cut;
      } else if( after_cut.empty() ) {
        img = before_cut;
      }
    }
  }
}

int main() {
  string path;
  cout << "Please enter the image path: ";
  cin >> path;
  Mat_<Vec3b> image = imread( path, IMREAD_COLOR );
  if( !image.data ) {
    cerr << "Image not loaded\n"; return -1;
  }
  string direction;
  cout << "Please enter the seam direction (0 for vertical, 1 for horizontal): ";
  cin >> direction;
  SeamDirection seam_direction;
  if ( direction == "0" || direction == "1" ) {
    if ( direction == "0" ) {
      seam_direction = VERTICAL;
    }
    else if ( direction == "1" ) {
      seam_direction = HORIZONTAL;
    }
  }
  else {
    cout << "Invalid choice, please re-run and try again" << endl;
    return 0;
  }

  string times;
  cout << "Number of times the image should be shrinked: ";
  cin >> times;
  int no_seams_to_remove = stoi( times );
  if( seam_direction == VERTICAL && no_seams_to_remove > image.cols ) {
    cerr << "The number of times is bigger than the number of columns.\n"; return -1;
  } else if ( seam_direction == HORIZONTAL && no_seams_to_remove > image.rows ) {
    cerr << "The number of times is bigger than the number of rows.\n"; return -1;
  }

  namedWindow( "Loaded Image", WINDOW_AUTOSIZE ); imshow( "Loaded Image", image );
  Mat_<Vec3b> retargeted = image.clone();

  /// Apply the algorithm
  for( int i = 0; i < no_seams_to_remove; ++i ) {
    Mat_<double> energy_image = makeEnergyImage( retargeted );
    Mat_<double> cumulative_energy_map = makeCumulativeEnergyMap( energy_image );

    /// Show energy image and cumulative energy map at first step
    if( i == 0 ) {
      namedWindow( "Energy Map at first step", WINDOW_AUTOSIZE ); imshow( "Energy Map at first step", energy_image );
      Mat color_cumulative_energy_map;
      double Cmin, Cmax;
      minMaxLoc(cumulative_energy_map, &Cmin, &Cmax);
      float scale = 255.0 / (Cmax - Cmin);
      cumulative_energy_map.convertTo(color_cumulative_energy_map, CV_8UC1, scale);
      applyColorMap(color_cumulative_energy_map, color_cumulative_energy_map, COLORMAP_JET);
      namedWindow("Cumulative Energy Map at first step", WINDOW_AUTOSIZE); imshow("Cumulative Energy Map at first step", color_cumulative_energy_map);
    }
    removeSeam( retargeted, cumulative_energy_map, seam_direction );
  }
  namedWindow( "Retargeted Image", WINDOW_AUTOSIZE ); imshow( "Retargeted Image", retargeted ); waitKey(0);

  return 0;
}
