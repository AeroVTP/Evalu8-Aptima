//=======================================================================================
// Name        : Evalu8.cpp
// Author      : Vidur Prasad
// Version     : 1.0.1
// Copyright   : APTIMA Inc.
// Description : Evaluate drone footage to asses visual clutter based load
// Error Messages : 1 --> Image cannot be read or is empty
//========================================================================================

//========================================================================================
// Metrics Polled
// 		Basic Number of Features --> SURF Detection 
//		Number of Corners 1 --> Harris Corner Detection
//		Number of Corners 2 --> Shi-Tomasi Feature Extraction
//      Number of Edges --> Canny Edge Detector
//		Optical Flow Analysis --> Farneback Dense Optical Flow Analysis
//=======================================================================================

//include opencv library files
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <opencv2/nonfree/ocl.hpp>

//include c++ files
#include <iostream>
#include <fstream>
#include <ctime>
#include <time.h>
#include <thread>         
#include <chrono>         
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <math.h>
#include <algorithm>
#include <vector>

template <typename T, size_t N> const T* mybegin(const T (&a)[N]) { return a; }
template <typename T, size_t N> const T* myend  (const T (&a)[N]) { return a+N; }

//namespaces for convenience
using namespace cv;
using namespace std;

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
                    double, const Scalar& color)
{
	flow *= 1.5;
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

long double computeOpticalFlowAnalysis(/*vector <Mat> framesForOFA,*/ Mat prevFrame, Mat currFrame,  int i, int FRAME_HEIGHT, int FRAME_WIDTH)
{
	////start optical flow analysis////
	////farneback dense optical flow analysis////
	Mat gray;
	Mat prevgray;
	Mat flow;
	Mat cflow;

	cvtColor(currFrame, gray,COLOR_BGR2GRAY);
	cvtColor(prevFrame, prevgray, COLOR_BGR2GRAY);

	imshow("GrayScale Image", gray);

	calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

	cvtColor(prevgray, cflow, COLOR_GRAY2BGR);

	flow *= 1;

	drawOptFlowMap(flow, cflow, 15, 1.5, Scalar(0, 255, 0));
	imshow("flow", cflow);
	return (abs(sum(flow)[0])) / (FRAME_HEIGHT * FRAME_WIDTH) * 1000;
	//return  abs((mean(flow).val[0] * 10000));
}

//method that returns date and time as a string to tag txt files
const string currentDateTime() 
{	
	//creating time object that reads current time
    time_t now = time(0);

    //creating time strucutre
    struct tm tstruct;

    //creating a character buffer of 80 characters
    char buf[80];

    //checking urrent local time
    tstruct = *localtime(&now);

    //writing time to string
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    //returning the string with the time
    return buf;
}

int computeSURF(vector <Mat> frames, int i)
{
	////start SURF extraction////

	//setting constant integerr minimum Hessian for SURF Recommended between 400-800
	const int minHessian = 500;

	//running SURF detector
	SurfFeatureDetector detector(minHessian);

	vector <KeyPoint> vectKeyPoints;

	Mat matFrameKeyPoints;

    detector.detect( frames.at(i), vectKeyPoints );

    drawKeypoints( frames.at(i), vectKeyPoints, matFrameKeyPoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

    imshow("SURF Detection", matFrameKeyPoints );

    return vectKeyPoints.size();
    ////end SURF extraction////
}

int computeHarris(vector <Mat> grayFrames, vector <Mat> frames, int i)
{
	////start Harris feature extraction

	int numberOfHarrisCornersCounter = 0;

	int blockSize = 3;

	//setting aperture size to constant of three
	const int apertureSize = 3;

	//initializing variables for Harris & Shi Tomasi min & max values
	double harrisMinValue;
	double harrisMaxValue;
	double harrisQualityLevel = 35;
	double maxQualityLevel = 100;

    //read in frame formatted for use in Harris
    Mat harrisDST = Mat::zeros(grayFrames.at(i).size(), CV_32FC(6) );

    //read in frame formatted for use in Harris
    Mat mc = Mat::zeros( grayFrames.at(i).size(), CV_32FC1 );

    //run Corner Eigen Vals and Vecs to find corners
    cornerEigenValsAndVecs( grayFrames.at(i), harrisDST, blockSize, apertureSize, BORDER_DEFAULT );

    //use Eigen values to step through each pixel individaully and finish applying equation
    for( int j = 0; j < grayFrames.at(i).rows; j++ )
    {
    	//for each column
    	for( int h = 0; h < grayFrames.at(i).cols; h++ )
    	{
    		//apply algorithm
			float lambda_1 = harrisDST.at<Vec6f>(j, h)[0];
			float lambda_2 = harrisDST.at<Vec6f>(j, h)[1];
			mc.at<float>(j,h) = lambda_1*lambda_2 - 0.04f*pow( ( lambda_1 + lambda_2 ), 2 );
    	}
    }

    //find locations of minimums and maximums 
    minMaxLoc( mc, &harrisMinValue, &harrisMaxValue, 0, 0, Mat() );

    //save frame in temporary matrix 
    Mat harrisCornerFrame = frames.at(i);

    //apply harris properly to every pixel
    for( int j = 0; j < grayFrames.at(i).rows; j++ )
    {
    	//for each column
    	for( int h = 0; h < grayFrames.at(i).cols; h++ )
	    {
			if( mc.at<float>(j,h) > harrisMinValue + ( harrisMaxValue - harrisMinValue )* harrisQualityLevel/maxQualityLevel)
			{
				//apply algorithm, and increment counters
				numberOfHarrisCornersCounter++;
			}
		}

	}

	return numberOfHarrisCornersCounter;
}

int computeShiTomasi(vector <Mat> grayFrames, int i)
{	
	//corners 
	vector<Point2f> cornersf;

	//setting quality level
	const double qualityLevel = 0.1;

	//setting minium distance between points
	const double minDistance = 10;

	//setting block size to search for
	const int blockSize = 3;

	//will use Harris detector seperately
	const bool useHarrisDetector = false;

	//setting k constant value
	const double k = 0.04;

	//setting max number of corners to largest possible value
	const int maxCorners = numeric_limits<int>::max();

	//perform Shi-Tomasi algorithm
    goodFeaturesToTrack(grayFrames.at(i), cornersf, maxCorners, qualityLevel, minDistance, 
    Mat(), blockSize, useHarrisDetector,k);

    return cornersf.size();

    ////end Shi-Tomasi feature extraction
}

int computeCanny(vector <Mat> frames, int i)
{	
	vector<Vec4i> hierarchy;

	typedef vector<vector<Point> > TContours;

	TContours contours;

	Mat cannyFrame; 

	 ////start Canny detector
	Canny(frames.at(i), cannyFrame, 115, 115);

	findContours(cannyFrame, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

	imshow("Canny Edges", cannyFrame);
    ////end Canny detector

    return contours.size();
}

long int computeFinalScore(Vector <int> vectNumberOfKeyPoints,Vector <int> numberOfHarrisCorners,
	Vector <int> numberOfShiTomasiKeyPoints, Vector <int> numberOfContours, long int opticalFlowAnalysisFarnebackNumbers, int i)
{
	double ratioNumKeyPoints;
	double ratioShiTomasi;
	double ratioHarrisCorners;
	double ratioContours;

	if (abs(*max_element((vectNumberOfKeyPoints.begin()), (vectNumberOfKeyPoints.end()))/75) == 0)
	{	ratioNumKeyPoints = 1;	}
	else
	{	ratioNumKeyPoints = abs(*max_element((vectNumberOfKeyPoints.begin()), (vectNumberOfKeyPoints.end())) / 75); }

	if (abs(*max_element((numberOfShiTomasiKeyPoints.begin()), (numberOfShiTomasiKeyPoints.end()))/75) == 0)
	{	ratioShiTomasi = 1;	}
	else
	{	ratioShiTomasi = abs(*max_element((numberOfShiTomasiKeyPoints.begin()), (numberOfShiTomasiKeyPoints.end())) / 75); }

	if (abs(*max_element((numberOfHarrisCorners.begin()), (numberOfHarrisCorners.end())) / 75) == 0)
	{	ratioHarrisCorners = 1;	}
	else
	{	ratioHarrisCorners = abs(*max_element((numberOfHarrisCorners.begin()), (numberOfHarrisCorners.end())) / 75); }

	if (abs(*max_element((numberOfContours.begin()), (numberOfContours.end())) / 75) == 0)
	{	ratioContours = 1;	}
	else
	{	ratioContours = abs(*max_element((numberOfContours.begin()), (numberOfContours.end())) / 75); }

	double numberOfKeyPointsNormalized = abs(vectNumberOfKeyPoints[i] / ratioNumKeyPoints);
	double numberOfHarrisCornersNormalized = abs(numberOfHarrisCorners[i] / ratioHarrisCorners);
	double numberOfShiTomasiKeyPointsNormalized = abs(numberOfShiTomasiKeyPoints[i] / ratioShiTomasi);
	double numberOfContoursNormalized = abs(numberOfContours[i] / ratioContours);

	opticalFlowAnalysisFarnebackNumbers = abs(opticalFlowAnalysisFarnebackNumbers);

	if(opticalFlowAnalysisFarnebackNumbers <= 250)
	{
		opticalFlowAnalysisFarnebackNumbers = 0;
	}

	else if(opticalFlowAnalysisFarnebackNumbers <= 500)
	{
		opticalFlowAnalysisFarnebackNumbers = 25;
	}

	else if(opticalFlowAnalysisFarnebackNumbers <= 2000)
	{
		opticalFlowAnalysisFarnebackNumbers = 75;
	}

	else
	{
		opticalFlowAnalysisFarnebackNumbers = 100;
	}

	long int finalScore = abs(((numberOfKeyPointsNormalized + numberOfShiTomasiKeyPointsNormalized +
			numberOfHarrisCornersNormalized + numberOfContoursNormalized + opticalFlowAnalysisFarnebackNumbers) / 5)
				);

	if (abs(finalScore) >= 500)
	{
		cout << "Final Rating Over 500" << endl;
		return -1;
	}
	return finalScore;

	/*return ((vectNumberOfKeyPoints[i] / 4) + (numberOfHarrisCorners[i] / 1.87) +
					(numberOfShiTomasiKeyPoints[i] /.8) + (numberOfContours[i] / 20) +
					(opticalFlowAnalysisFarnebackNumbers[i-4] / 200000)) / 5;*/
}

String convertToString(int value)
{
	//initiating conversion string stream
	ostringstream convert;

	//counting number of corners and sending to converter stream
	convert << value;

	//reading number of corners into string from converter
	string returnString = convert.str();

    //concatanating strings to presentable format
    return returnString;


}

void frameHasData(Mat *frameToBeDisplayed, int error1)
{
	//if the frame is empty
	if (frameToBeDisplayed->empty())
	{
		//Error Message 1, Image is empty
		cout << "Error: 1 --> Image is empty, cannot be loaded" << endl;

		//throw an error
		throw "Error: 1 --> Image is empty, cannot be loaded";

		//add that error did occur
		error1++;

		exit(-1);
	}

}

void destroyWindows()
{


	//close window
	destroyWindow("Raw");
	destroyWindow("SURF Detection");
	destroyWindow("Harris Corner Detection");
	destroyWindow("Shi-Tomasi Corner Detection");
	destroyWindow("Farneback Dense Optical Flow Analysis");
}

void saveToTxtFile(int FRAME_RATE, Vector <int> vectNumberOfKeyPoints, Vector <int>numberOfShiTomasiKeyPoints, Vector <int>
numberOfContours, Vector <int> numberOfHarrisCorners, Vector <long double> opticalFlowAnalysisFarnebackNumbers, const char* filename)
{
	ofstream file;

	//creating filename ending
	string vectFilenameAppend = " vectNumberOfKeyPoints.txt";

		//concanating and creating file name string
	string strVectFilename = filename + currentDateTime() + vectFilenameAppend;

	file.open (strVectFilename);

	//save txt file
	for(int v = 0; v < vectNumberOfKeyPoints.size() - 5; v++)
	{
		file << "Frame Number " << v << " at " << (v * (1.0 / FRAME_RATE)) << " seconds has ";
		file << vectNumberOfKeyPoints[v];
		file << " SURF key points & ";
		file << numberOfShiTomasiKeyPoints[v];
		file << " Shi-Tomasi key points & ";
		file << " & " << numberOfContours[v] << " contours & ";
		file << numberOfHarrisCorners[v];
		file << "Harris Corners.\n";
	}

	//close file stream
	file.close();
}

void computeRunTime(clock_t t1, clock_t t2, int framesRead)
{
	//subtract from start time
	float diff ((float)t2-(float)t1);

	//calculate frames per second
	double frameRateProcessing = framesRead / diff;

	//display amount of time for run time
	cout << diff << " seconds of run time." << endl;

	//display number of frames processed per second
	cout << frameRateProcessing << " frames processed per second." << endl;

	cout << framesRead << " frames read." << endl;
}

void writeInitialStats(int NUMBER_OF_FRAMES, int FRAME_RATE, int FRAME_WIDTH, int FRAME_HEIGHT, const char* filename)
{
	////writing stats to txt file
	//initiating write stream
	ofstream writeToFile;

	//creating filename ending
	string filenameAppend = "Stats.txt";

	//concanating and creating file name string
	string strFilename = filename + currentDateTime() + filenameAppend;

	//open file stream and begin writing file
	writeToFile.open (strFilename);

	//write video statistics
	writeToFile << "Stats on video >> There are = " << NUMBER_OF_FRAMES << " frames. The frame rate is " << FRAME_RATE
	<< " frames per second. Resolution is " << FRAME_WIDTH << " X " << FRAME_HEIGHT;

	//close file stream
	writeToFile.close();

	//display video statistics
	cout << "Stats on video >> There are = " << NUMBER_OF_FRAMES << " frames. The frame rate is " << FRAME_RATE
	<< " frames per second. Resolution is " << FRAME_WIDTH << " X " << FRAME_HEIGHT << endl;;

}

//main method
int main() {

	//creating initial and final clock objects
	//taking current time when run starts
	clock_t t1=clock();
	clock_t t3, t4;

	//random number generator
	RNG rng(12345);
    
	//setting constant filename to read form
	const char* filename = "assets/P2-T1-V1-TCheck-Final.mp4";
	//const char* filename = "assets/The Original Grumpy Cat!.mp4";
	//const char* filename = "assets/P8_T5_V1.mp4";
	//defining VideoCapture object and filename to capture from

	VideoCapture capture(filename);

	//matrix of frame for shi Tomasi
	Mat shiTomasiFrame;

	Mat placeHolder = Mat::eye(1, 2, CV_64F);

	//declaring strings for all metrics
    string strRating, strNumberOfHarrisCorners, strNumberOfShiTomasiCorners, numberOfKeyPointsSURF, strCanny, strActiveTimeDifference;

    //initializing string to display blank
	string strDisplay =  "";
	string strNumberOpticalFlowAnalysis = "";

	//creating a vector of matricies of all frames
	vector <Mat> frames;
	//creating a vector of matricies of all grayscaleframes
	vector <Mat> grayFrames;

	//creating vector of pointers for OFA to avoid memory leaks
	vector <Mat*> framesForOFA;

	//creating vectors to store all metrics
	vector <int> numberOfContours;
	vector <int> numberOfShiTomasiKeyPoints;
	vector <int> numberOfHarrisCorners;
	vector <long double> opticalFlowAnalysisFarnebackNumbers;
	vector <int> vectNumberOfKeyPoints;

	//collecting statistics about the video
	//constants that will not change
	const int NUMBER_OF_FRAMES =(int) capture.get(CV_CAP_PROP_FRAME_COUNT);
	const int FRAME_RATE = (int) capture.get(CV_CAP_PROP_FPS);
	const int FRAME_WIDTH = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	const int FRAME_HEIGHT = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

	writeInitialStats(NUMBER_OF_FRAMES, FRAME_RATE, FRAME_WIDTH, FRAME_HEIGHT, filename);

	// declaring and initially setting variables that will be actively updated during runtime
	int framesRead = (int) capture.get(CV_CAP_PROP_POS_FRAMES);
	double framesTimeLeft = (capture.get(CV_CAP_PROP_POS_MSEC)) / 1000;

	//display welcome image
	imshow("Welcome", imread("assets/Aptima.jpg"));

	//put thread to sleep until user is ready
  	this_thread::sleep_for (std::chrono::seconds(5));

   	//close welcome image
   	destroyWindow("Welcome");

   	//initializing counters
   	int generalDebugCounter = 0;
	int q = 0;
	int i = 0;
	int error1 = 0;
	int prevFramesRead = 0;
	int prevFramesReadCounter = 0;

	//actual run time, while video is not finished
	while(framesRead < NUMBER_OF_FRAMES)
	{
		//create pointer to new object
		Mat * frameToBeDisplayed = new Mat();

		//reading in current frame
		capture.read(*frameToBeDisplayed);

		//check if frame has data
		frameHasData(frameToBeDisplayed, error1);
		
		//adding current frame to vector/array list of matricies
		frames.push_back(*frameToBeDisplayed);

		//convert Shi Tomasi frame to grayscale
		cvtColor(frames.at(i), shiTomasiFrame, CV_BGR2GRAY);

		grayFrames.push_back(shiTomasiFrame);

		//imshow("Raw Frame", frames.at(i));

		//compute SURF
		vectNumberOfKeyPoints.push_back(computeSURF(frames, i));
		String numberOfKeyPointsSURF = convertToString(vectNumberOfKeyPoints.at(i));

		//compute Harris
		numberOfHarrisCorners.push_back(computeHarris(grayFrames, frames, i));
		String strNumberOfHarrisCorners = convertToString(numberOfHarrisCorners.at(i));

		//compute ShiTomasi
		numberOfShiTomasiKeyPoints.push_back(computeShiTomasi(grayFrames, i));
		String strNumberOfShiTomasiCorners = convertToString(numberOfShiTomasiKeyPoints.at(i));

		//compute Canny
		numberOfContours.push_back(computeCanny(frames, i));
		strCanny = convertToString(numberOfContours.at(i));

		//if ready for OFA
		if(i > 10)
		{
			//compute FDOFA
			opticalFlowAnalysisFarnebackNumbers.push_back(computeOpticalFlowAnalysis(*framesForOFA.at(q-1), *framesForOFA.at(q-2), i, FRAME_HEIGHT, FRAME_WIDTH));
			strNumberOpticalFlowAnalysis = convertToString(computeOpticalFlowAnalysis(*framesForOFA.at(q-1),  *framesForOFA.at(q-2), i, FRAME_HEIGHT, FRAME_WIDTH));
			strRating = to_string(computeFinalScore(vectNumberOfKeyPoints, numberOfHarrisCorners, numberOfShiTomasiKeyPoints, numberOfContours,
					computeOpticalFlowAnalysis(*framesForOFA.at(q-1),  *framesForOFA.at(q-2), i, FRAME_HEIGHT, FRAME_WIDTH), i));
		}
		//if not enough data has been generated for optical flow
		else if(i > 0 && i <= 3)
		{

			//creating text to display
			strDisplay = "SURF Features: " + numberOfKeyPointsSURF + " Shi-Tomasi: " + strNumberOfShiTomasiCorners + " Harris: "
			+ strNumberOfHarrisCorners + " Canny: " + strCanny + " Frame Number: " + to_string(framesRead) +  " SFP: " + strActiveTimeDifference;

			//creating black empty image
			Mat pic = Mat::zeros(45,2250,CV_8UC3);

			//adding text to image
			putText(pic, strDisplay, cvPoint(30,30),CV_FONT_HERSHEY_SIMPLEX, 1.25, cvScalar(0,255,0), 1, CV_AA, false);

			//displaying image
			imshow("Stats", pic);

		}

		//gather real time statistics
		framesRead = (int) capture.get(CV_CAP_PROP_POS_FRAMES);
		framesTimeLeft = (capture.get(CV_CAP_PROP_POS_MSEC)) / 1000;

		//creating text to display
		strDisplay = "SURF Features: " + numberOfKeyPointsSURF + " Shi-Tomasi: " + strNumberOfShiTomasiCorners + " Harris: "
		+ strNumberOfHarrisCorners + + " Canny: " + strCanny + " FDOFA: " + strNumberOpticalFlowAnalysis +  " Frame Number: " +
		to_string(framesRead) +  " Rating: " + strRating;

		//creating black empty image
		Mat pic = Mat::zeros(45,1850,CV_8UC3);

		//adding text to image
		putText(pic, strDisplay, cvPoint(30,30),CV_FONT_HERSHEY_SIMPLEX, 1, cvScalar(0,255,0), 1, CV_AA, false);

		//displaying image
		imshow("Stats", pic);

		//read in current key press
		char c = cvWaitKey(33);

		//if escape key is pressed
		if(c==27)
		{
			//reset key listener
			c = cvWaitKey(33);

			//display warning
			cout << "Are you sure you want to exit?" << endl;
			
			//if key is pressed again, in rapid succession
			if(c==27)
			{
				//display exiting message
				cout << "Exiting" << endl;

				saveToTxtFile(FRAME_RATE, vectNumberOfKeyPoints, numberOfShiTomasiKeyPoints, numberOfContours, numberOfHarrisCorners, opticalFlowAnalysisFarnebackNumbers, filename);

				computeRunTime(t1, clock(), (int) capture.get(CV_CAP_PROP_POS_FRAMES));

				destroyWindows();

				//report file finished writing
				cout << "Finished writing file, Goodbye." << endl;

				//exit program
				return 0;
			}
		}

		//if this iteration of the while loop analyzed a new frame
		if(prevFramesRead != framesRead)
		{	
			prevFramesRead = framesRead;

			framesForOFA.push_back(frameToBeDisplayed);

			if(prevFramesReadCounter % 2 != 0)
			{
				//read current time
				t3=clock();

				//subtract from start time
				float activeTimeDifference ((float)t3-(float)t4);

				//convert milliseconds to seconds				
				activeTimeDifference *= 1000;

			}
			else
			{

				//read current time
				t4=clock();

				//subtract from start time
				float activeTimeDifference ((float)t3-(float)t4);

				//convert milliseconds to seconds				
				activeTimeDifference *= 1000;
			}

			//increment ready that next frame has been read
			prevFramesReadCounter++;
		}
		q++;
   		i++;

   		if(i > 5)
   		{
   			frames.at(i - 5) = placeHolder;
   			grayFrames.at(i - 5) = placeHolder;
   			delete framesForOFA.at(i - 5);
   		}

	}		

	for(int z = 0; z < framesForOFA.size(); z++)
	{
		delete framesForOFA.at(z);
	}

	frames.clear();
	grayFrames.clear();
	framesForOFA.clear();

	//print out debug and error information
	cout << "Debug Information >> i = " << i << "; Error 1 encountered " << error1 << "times. " << endl;

	saveToTxtFile(FRAME_RATE, vectNumberOfKeyPoints, numberOfShiTomasiKeyPoints, numberOfContours, numberOfHarrisCorners, opticalFlowAnalysisFarnebackNumbers, filename);

	computeRunTime(t1, clock(),(int) capture.get(CV_CAP_PROP_POS_FRAMES));

	//display finished, promt to close program
	cout << "Execution finished, file written, click to close window. " << endl;

	//wait for button press to proceed
	waitKey(0);

	destroyWindows();

	//return code is finished and ran successfully
	return 0;
}
