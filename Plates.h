// Plates.h
/// this header file is to detect the plate in the cars

#ifndef DETECT_PLATES_H
#define DETECT_PLATES_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include "Main.h"
#include "RecPlate.h"
#include "RecChar.h"
#include "Preprocess.h"
#include "Chars.h"

// global constants ///////////////////////////////////////////////////////////////////////////////
const double PLATE_WIDTH_PADDING_FACTOR = 1.3;
const double PLATE_HEIGHT_PADDING_FACTOR = 1.5;

// function prototypes ////////////////////////////////////////////////////////////////////////////
std::vector<PossiblePlate> detectPlatesInScene(cv::Mat &imgOriginalScene);

std::vector<PossibleChar> findPossibleCharsInScene(cv::Mat &imgThresh);

PossiblePlate extractPlate(cv::Mat &imgOriginal, std::vector<PossibleChar> &vectorOfMatchingChars);


# endif	// DETECT_PLATES_H

