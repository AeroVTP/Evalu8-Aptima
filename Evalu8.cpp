//=======================================================================================
// Name        : Evalu8.cpp
// Author      : Vidur Prasad
// Version     : 1.3.0
// Copyright   : APTIMA Inc.
// Description : Evaluate drone footage to asses visual clutter based load
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
#include <pthread.h>
#include <cstdlib>

//declaring templates for use in max element function
template <typename T, size_t N> const T* mybegin(const T (&a)[N]) { return a; }
template <typename T, size_t N> const T* myend  (const T (&a)[N]) { return a+N; }

//namespaces for convenience
using namespace cv;
using namespace std;

//multithreading global variables
static vector <Mat> globalFrames;
static vector <Mat> globalGrayFrames;
int FRAME_HEIGHT;
int FRAME_WIDTH;
//setting constant filename to read form
const char* filename = "assets/P2-T1-V1-TCheck-Final.mp4";
//const char* filename = "assets/The Original Grumpy Cat!.mp4";
//const char* filename = "assets/P8_T5_V1.mp4";

//SURF Global Variables
static Mat surfFrame;
int surfThreadCompletion = 0;
int numberOfSURFFeatures = 0;

//canny Global Variables
int numberOfContoursThread = 0;
int cannyThreadCompletion = 0;

//Shi Tomasi Global Variables
int shiTomasiFeatures = 0;
int shiTomasiThreadCompletion = 0;

//harris global variables
int numberOfHarrisCornersCounter = 0;
int harrisCornersThreadCompletion = 0;

//optical flow global variables
int sumOpticalFlow = 0;
int opticalFlowThreadCompletion = 0;

struct thread_data{
   int i;
};

//method that returns date and time as a string to tag txt files
const string currentDateTime()
{
	//creating time object that reads current time
    time_t now = time(0);
    //creating time structure
    struct tm tstruct;
    //creating a character buffer of 80 characters
    char buf[80];
    //checking current local time
    tstruct = *localtime(&now);
    //writing time to string
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    //returning the string with the time
    return buf;
}

//method to perform optical flow analysis
void *computeOpticalFlowAnalysisThread(void *threadarg)
{
	struct thread_data *data;
	data = (struct thread_data *) threadarg;
	int i = data->i;

	Mat prevFrame, currFrame;
	Mat gray, prevGray, flow,cflow;

	prevFrame = globalFrames.at(i-1);
	currFrame = globalFrames.at(i);

	//converting to grayscale
	cvtColor(currFrame, gray,COLOR_BGR2GRAY);
	cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

	//calculating optical flow
	calcOpticalFlowFarneback(prevGray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

	flow *= 1.5;
	//returning per pixel average movemet,iwth amplitude increase
	sumOpticalFlow = (abs(sum(flow)[0]));
	//sumOpticalFlow = ceil((abs(sum(flow)[0])) / (FRAME_HEIGHT * FRAME_WIDTH) * 1000);

	opticalFlowThreadCompletion = 1;
}

//calculate number of Harris corners
void *computeHarrisThread(void *threadarg)
{
	struct thread_data *data;
	data = (struct thread_data *) threadarg;
	int i = data->i;

	numberOfHarrisCornersCounter = 0;
	int blockSize = 3;
	const int apertureSize = 3;
	double harrisMinValue;
	double harrisMaxValue;
	double harrisQualityLevel = 35;
	double maxQualityLevel = 100;

    //create frame formatted for use in Harris
    Mat harrisDST = Mat::zeros(globalGrayFrames.at(i).size(), CV_32FC(6) );
    Mat mc = Mat::zeros( globalGrayFrames.at(i).size(), CV_32FC1 );
    Mat harrisCornerFrame = globalFrames.at(i);

    //run Corner Eigen Vals and Vecs to find corners
    cornerEigenValsAndVecs( globalGrayFrames.at(i), harrisDST, blockSize, apertureSize, BORDER_DEFAULT );

    //use Eigen values to step through each pixel individaully and finish applying equation
    for( int j = 0; j < globalGrayFrames.at(i).rows; j++ )
    {
    	for( int h = 0; h < globalGrayFrames.at(i).cols; h++ )
    	{
    		//apply algorithm
			float lambda_1 = harrisDST.at<Vec6f>(j, h)[0];
			float lambda_2 = harrisDST.at<Vec6f>(j, h)[1];
			mc.at<float>(j,h) = lambda_1*lambda_2 - 0.04f*pow( ( lambda_1 + lambda_2 ), 2 );
    	}
    }

    //find locations of minimums and maximums
    minMaxLoc( mc, &harrisMinValue, &harrisMaxValue, 0, 0, Mat() );

    //apply harris properly to every pixel
    for( int j = 0; j < globalGrayFrames.at(i).rows; j++ )
    {
    	for( int h = 0; h < globalGrayFrames.at(i).cols; h++ )
	    {
			if( mc.at<float>(j,h) > harrisMinValue + ( harrisMaxValue - harrisMinValue )* harrisQualityLevel/maxQualityLevel)
			{
				//apply algorithm, and increment counters
				numberOfHarrisCornersCounter++;
			}
		}
	}
    //if harris glitch occurs
	if(numberOfHarrisCornersCounter > 10000)
	{
		numberOfHarrisCornersCounter = 2000;
	}

	harrisCornersThreadCompletion = 1;
}

//calculate number of Shi-Tomasi corners
void *computeShiTomasiThread(void *threadarg)
{
	struct thread_data *data;
    data = (struct thread_data *) threadarg;
    int i = data->i;
	vector<Point2f> cornersf;
	const double qualityLevel = 0.1;
	const double minDistance = 10;
	const int blockSize = 3;
	const double k = 0.04;

	//harris detector is used seperately
	const bool useHarrisDetector = false;

	//setting max number of corners to largest possible value
	const int maxCorners = numeric_limits<int>::max();

	//perform Shi-Tomasi algorithm
    goodFeaturesToTrack(globalGrayFrames.at(i), cornersf, maxCorners, qualityLevel, minDistance,
    Mat(), blockSize, useHarrisDetector,k);

    //return number of Shi Tomasi corners
    shiTomasiFeatures = cornersf.size();

    shiTomasiThreadCompletion = 1;
}


void *computeSURFThread(void *threadarg)
{
   struct thread_data *data;
   data = (struct thread_data *) threadarg;
   int i = data->i;
   globalFrames.at(i).copyTo(surfFrame);
   //setting constant integer minimum Hessian for SURF Recommended between 400-800
   const int minHessian = 500;
   vector <KeyPoint> vectKeyPoints;
   //running SURF detector
   SurfFeatureDetector detector(minHessian);
   detector.detect(surfFrame, vectKeyPoints );
   //drawing keypoints
   drawKeypoints(surfFrame, vectKeyPoints, surfFrame, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
   numberOfSURFFeatures = vectKeyPoints.size();
   surfThreadCompletion = 1;
   pthread_exit(NULL);
}

//calculate number of contours
void *computeCannyThread(void *threadarg)
{
	Mat cannyFrame;
	vector<Vec4i> hierarchy;
	typedef vector<vector<Point> > TContours;
	TContours contours;
	struct thread_data *data;
	data = (struct thread_data *) threadarg;
	int i = data->i;
	//run canny edge detector
	Canny(globalFrames.at(i), cannyFrame, 115, 115);
	findContours(cannyFrame, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	//return number of contours detected
	//imshow("globalFrames", contours);

    numberOfContoursThread = contours.size();

    cannyThreadCompletion = 1;
}

//method to normalize normal values
double normalizeValues(double value, double meanValue)
{
	//deterine normalization factor to 50
	meanValue =  50 / meanValue;

	//apply normalization ration
	value *= meanValue;

	//give extra weight to differential from 50 & add 50 to ensure positive number
	value = abs(((value - 50) * 1.25)) + 50;

	return value;
}

//method to normalize OFA values
double normalizeOFAValues(double value, double meanValue)
{
	//deterine normalization factor to 50
	meanValue =  50 / meanValue;

	//apply normalization ration
	value *= meanValue;

	//give extra weight to differential from 50 & add 50 to ensure positive number

	value = abs(((value - 50) * 1.25)) + 50;

	//if extreme motion
	if(value > 10000)
	{
		//take natural log, after log base 10, and then multiply by 20;
		return (log(log10(value)) * 20);
	}

	else
	{
		return value;
	}

}

//calculate log of a vector and normalize
double calcLogVector(double value)
{
	return log10(value) + 500;
}

//calculate mean of vector of ints
double calculateMeanVector(Vector <int> scores)
{
  double total;
  //sum all elements of vector
  for(int i = 0; i < scores.size(); i++)
  { total += abs(scores[i]); }
  //divide by number of elements
  return total / scores.size();
}

//calculate mean of vector of ints
double calculateMeanVector(vector <int> scores)
{
  double total;
  //sum all elements of vector
  for(int i = 0; i < scores.size(); i++)
  { total += abs(scores.at(i)); }
  //divide by number of elements
  return total / scores.size();
}


//calculate mean of vector of oubles
double calculateMeanVector(Vector<double> scores)
{
  double total;
  //sum all elements of vector
  for(int i = 0; i < scores.size(); i++)
  { total += abs(scores[i]); }
  //divide by number of elements
  return total / scores.size();
}

//method to compute final score
int computeFinalScore(Vector <int> vectNumberOfKeyPoints,Vector <int> numberOfHarrisCorners,
	Vector <int> numberOfShiTomasiKeyPoints, Vector <int> numberOfContours, Vector <double> opticalFlowAnalysisFarnebackNumbers, int i)
{
	//normalize values using mean
	double numberOfKeyPointsNormalized = 3 * normalizeValues(vectNumberOfKeyPoints[i], calculateMeanVector(vectNumberOfKeyPoints));
	double numberOfHarrisCornersNormalized = 3 * normalizeValues(numberOfHarrisCorners[i], calculateMeanVector(numberOfHarrisCorners));
	double numberOfShiTomasiKeyPointsNormalized = 3 * normalizeValues(numberOfShiTomasiKeyPoints[i], calculateMeanVector(numberOfShiTomasiKeyPoints));
	double numberOfContoursNormalized = normalizeValues(numberOfContours[i], calculateMeanVector(numberOfContours));
	double opticalFlowAnalysisFarnebackNumbersNormalized = 10 * normalizeOFAValues(opticalFlowAnalysisFarnebackNumbers[i], calculateMeanVector(opticalFlowAnalysisFarnebackNumbers));

	//determine final score by averaging
	long int finalScore = abs(((numberOfKeyPointsNormalized + numberOfShiTomasiKeyPointsNormalized +
			numberOfHarrisCornersNormalized + numberOfContoursNormalized + opticalFlowAnalysisFarnebackNumbersNormalized) / 20)
				);

	//if value is not possible, report error
	if (abs(finalScore) > 100 || finalScore < 0)
	{
		finalScore = -101;
	}
	return finalScore;
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

void saveToTxtFile(int FRAME_RATE, Vector <int> vectNumberOfKeyPoints, Vector <int> numberOfShiTomasiKeyPoints, Vector <int>
numberOfContours, Vector <int> numberOfHarrisCorners, Vector <double> opticalFlowAnalysisFarnebackNumbers, const char* filename)
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

	ofstream fileStreamTwo;
	//creating filename ending
	string vectFilenameAppendTwo = " meanData.ev8";
		//concanating and creating file name string
	string strVectFilenameTwo = filename + currentDateTime() + vectFilenameAppendTwo;
	file.open (strVectFilename);
	//save txt file
	for(int v = 0; v < vectNumberOfKeyPoints.size() - 5; v++)
	{
		file << calculateMeanVector(vectNumberOfKeyPoints) << ";" << endl;
		file << calculateMeanVector(numberOfShiTomasiKeyPoints) << ";" << endl;
		file << calculateMeanVector(numberOfContours) << ";" << endl;
		file << calculateMeanVector(numberOfHarrisCorners) << ";" << endl;
		file << calculateMeanVector(opticalFlowAnalysisFarnebackNumbers) << ";" << endl;

	}

	//close file stream
	fileStreamTwo.close();
}

void saveToTxtFile(vector <int> finalRatings)
{
	ofstream file;

	//creating filename ending
	string vectFilenameAppend = "finalRatings.txt";

		//concanating and creating file name string
	string strVectFilename = filename + currentDateTime() + vectFilenameAppend;

	file.open (strVectFilename);

	//save txt file
	for(int v = 0; v < finalRatings.size() ; v++)
	{
		file << v << " " << finalRatings.at(v) << endl;
	}

	//close file stream
	file.close();
}

//method to calculate tootal run time
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

//calculate time for each iteration
double calculateFPS(clock_t tStart, clock_t tFinal)
{
	return 1/((((float)tFinal-(float)tStart) / CLOCKS_PER_SEC));
}

//write initial statistics about the video
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

void normalizeRatings(vector<int> finalScores)
{
	vector <int> normalizedFinalScores;
	double meanOfVector = calculateMeanVector(finalScores);

	//int maxElement = max_element(begin(finalScores), end(finalScores));
	double maxElement = *max_element(finalScores.begin(), finalScores.end());
	double minElement = *min_element(finalScores.begin(), finalScores.end());

	for(int i = 0; i < finalScores.size(); i++)
	{
		normalizedFinalScores.push_back(((finalScores.at(i) - minElement) / (maxElement - minElement))*100);
		cout << ((finalScores.at(i) - minElement) / (maxElement - minElement))*100 << endl;
	}

	saveToTxtFile(normalizedFinalScores);
}


//main method
int main() {

	//creating initial and final clock objects
	//taking current time when run starts
	clock_t t1=clock();
	clock_t t3, t4;

	//random number generator
	RNG rng(12345);
    

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

	vector <double> logOpticalFlowAnalysisFarnebackNumbers;

	//creating vectors to store all metrics
	vector <int> numberOfContours;
	vector <int> numberOfShiTomasiKeyPoints;
	vector <int> numberOfHarrisCorners;
	vector <double> opticalFlowAnalysisFarnebackNumbers;
	vector <int> vectNumberOfKeyPoints;

	vector <int> finalScores;

	//collecting statistics about the video
	//constants that will not change
	const int NUMBER_OF_FRAMES =(int) capture.get(CV_CAP_PROP_FRAME_COUNT);
	const int FRAME_RATE = (int) capture.get(CV_CAP_PROP_FPS);
	FRAME_WIDTH = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	FRAME_HEIGHT = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

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
	int q = 0;
	int i = 0;
	int prevFramesRead = 0;
	int prevFramesReadCounter = 0;

	//actual run time, while video is not finished
	while(framesRead < NUMBER_OF_FRAMES)
	{
		clock_t tStart = clock();

		//create pointer to new object
		Mat * frameToBeDisplayed = new Mat();

		//reading in current frame
		capture.read(*frameToBeDisplayed);
		
		//adding current frame to vector/array list of matricies
		frames.push_back(*frameToBeDisplayed);

		globalFrames.push_back(*frameToBeDisplayed);

		//convert Shi Tomasi frame to grayscale
		cvtColor(frames.at(i), shiTomasiFrame, CV_BGR2GRAY);

		grayFrames.push_back(shiTomasiFrame);
		globalGrayFrames.push_back(shiTomasiFrame);

		pthread_t surfThread;
		struct thread_data surfThreadData;
	    surfThreadData.i = i;
	    int surfThreadRC = pthread_create(&surfThread, NULL, computeSURFThread, (void *)&surfThreadData);
	    if (surfThreadRC)
	    {
	    	cout << "Error:unable to create thread," << surfThreadRC << endl;
	    	exit(-1);
	    }

		pthread_t cannyThread;
		struct thread_data cannyThreadData;
		cannyThreadData.i = i;
	    int cannyThreadRC = pthread_create(&cannyThread, NULL, computeCannyThread, (void *)&cannyThreadData);

		if (cannyThreadRC)
		{
			cout << "Error:unable to create thread," << cannyThreadRC << endl;
			exit(-1);
		}

		pthread_t shiTomasiThread;
		struct thread_data shiTomasiData;
		shiTomasiData.i = i;
		int shiTomasiRC = pthread_create(&shiTomasiThread, NULL, computeShiTomasiThread, (void *)&shiTomasiData);

		if (shiTomasiRC)
		{
			cout << "Error:unable to create thread," << shiTomasiThread << endl;
			exit(-1);
		}

		pthread_t harrisThread;
		struct thread_data harrisData;
		harrisData.i = i;
		int harrisRC = pthread_create(&harrisThread, NULL, computeHarrisThread, (void *)&harrisData);

		if (harrisRC)
		{
			cout << "Error:unable to create thread," << harrisThread << endl;
			exit(-1);
		}

		if(i > 10)
		{
			pthread_t opticalFlowThread;
			struct thread_data opticalFlowData;
			opticalFlowData.i = i;
			int opticalFlowRC = pthread_create(&opticalFlowThread, NULL, computeOpticalFlowAnalysisThread, (void *)&opticalFlowData);

			if (opticalFlowRC)
			{
				cout << "Error:unable to create thread," << opticalFlowRC << endl;
				exit(-1);
			}
		}

		if(i<= 10)
		{
			while(surfThreadCompletion == 0 || cannyThreadCompletion == 0 || shiTomasiThreadCompletion == 0 || harrisCornersThreadCompletion == 0)
			{
			}
		}
		else
		{
			while(surfThreadCompletion == 0 || cannyThreadCompletion == 0 ||
								shiTomasiThreadCompletion == 0 || harrisCornersThreadCompletion == 0 || opticalFlowThreadCompletion == 0)
			{
			}
		}
		shiTomasiThreadCompletion = 0;
		surfThreadCompletion = 0;
		cannyThreadCompletion = 0;
		harrisCornersThreadCompletion = 0;
		opticalFlowThreadCompletion = 0;

		//write Canny
		numberOfContours.push_back(numberOfContoursThread);
		strCanny = to_string(numberOfContours.at(i));

		//write ShiTomasi
		numberOfShiTomasiKeyPoints.push_back(shiTomasiFeatures);
		String strNumberOfShiTomasiCorners = to_string(shiTomasiFeatures);

		//write SURF
		vectNumberOfKeyPoints.push_back(numberOfSURFFeatures);
		String numberOfKeyPointsSURF = to_string(numberOfSURFFeatures);

		//write Harris
		numberOfHarrisCorners.push_back(numberOfHarrisCornersCounter);
		String strNumberOfHarrisCorners = to_string(numberOfHarrisCornersCounter);

		//if ready for OFA
		if(i > 10)
		{
			opticalFlowAnalysisFarnebackNumbers.push_back(sumOpticalFlow);
			strNumberOpticalFlowAnalysis = to_string(sumOpticalFlow);
			//compute FDOFA
			//logOpticalFlowAnalysisFarnebackNumbers.push_back(calcLogVector(opticalFlowAnalysisFarnebackNumbers.at(i-11)));
			finalScores.push_back(computeFinalScore(vectNumberOfKeyPoints, numberOfHarrisCorners, numberOfShiTomasiKeyPoints, numberOfContours,
					opticalFlowAnalysisFarnebackNumbers, i));
			strRating = to_string(finalScores.at(i-11));
		}
		//if not enough data has been generated for optical flow
		else if(i > 0 && i <= 3)
		{

			//creating text to display
			strDisplay = "SURF Features: " + numberOfKeyPointsSURF + " Shi-Tomasi: " + strNumberOfShiTomasiCorners + " Harris: "
			+ strNumberOfHarrisCorners + " Canny: " + strCanny + " Frame Number: " + to_string(framesRead);

			//creating black empty image
			Mat pic = Mat::zeros(45,1910,CV_8UC3);

			//adding text to image
			putText(pic, strDisplay, cvPoint(30,30),CV_FONT_HERSHEY_SIMPLEX, 1.25, cvScalar(0,255,0), 1, CV_AA, false);

			//displaying image
			imshow("Stats", pic);

		}

		//gather real time statistics
		framesRead = (int) capture.get(CV_CAP_PROP_POS_FRAMES);
		framesTimeLeft = (capture.get(CV_CAP_PROP_POS_MSEC)) / 1000;

		clock_t tFinal = clock();

		strActiveTimeDifference = to_string(calculateFPS(tStart, tFinal));

		//creating text to display
		strDisplay = "SURF: " + numberOfKeyPointsSURF + " Shi-Tomasi: " + strNumberOfShiTomasiCorners + " Harris: "
		+ strNumberOfHarrisCorners + + " Canny: " + strCanny + " FDOFA: " + strNumberOpticalFlowAnalysis +  " Frame Number: " +
		to_string(framesRead) +  " Rating: " + strRating +  " FPS: " + strActiveTimeDifference.substr(0, 4);

		//creating black empty image
		Mat pic = Mat::zeros(45,1910,CV_8UC3);

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

				normalizeRatings(finalScores);

				saveToTxtFile(FRAME_RATE, vectNumberOfKeyPoints, numberOfShiTomasiKeyPoints, numberOfContours, numberOfHarrisCorners, logOpticalFlowAnalysisFarnebackNumbers, filename);

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
   			globalFrames.at(i - 5) = placeHolder;
   			globalGrayFrames.at(i-5) = placeHolder;
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

	normalizeRatings(finalScores);

	saveToTxtFile(FRAME_RATE, vectNumberOfKeyPoints, numberOfShiTomasiKeyPoints, numberOfContours, numberOfHarrisCorners, logOpticalFlowAnalysisFarnebackNumbers, filename);

	computeRunTime(t1, clock(),(int) capture.get(CV_CAP_PROP_POS_FRAMES));

	//display finished, promt to close program
	cout << "Execution finished, file written, click to close window. " << endl;

	//wait for button press to proceed
	waitKey(0);

	destroyWindows();

	//return code is finished and ran successfully
	return 0;
}
