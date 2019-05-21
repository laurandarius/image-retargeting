# image-retargeting
Implementation of the seam carving algorithm used for image retargeting.

## Getting started
These instructions will help you run your copy of the program, but they are targeting Mac users only.

### Prerequisites
* `opencv` installed.
* `pkg-config` installed.

### Installing
Steps to run the program on Mac:
1. Compile the program using: `g++ $(pkg-config --cflags --libs opencv4) -std=c++11  image-retargeting.cpp -o <BINARY_FILE_NAME>`
2. Run the program: <BINARY_FILE_NAME> <IMAGE_PATH> <NUMBER_OF_SEAMS_TO_REMOVE>

