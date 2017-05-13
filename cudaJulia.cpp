/*
 * Name: Aaron Marlowe
 * Date: 12/8/2016
 * Assignment: High Performance Final Project (HOST CODE)
 * Description: This program generates a 1216x1024 image of a julia fractal serially. This image can be manipulated 
 * 	        (once loaded) by pressing a variety of keys to change the image parameters (instructions will be on
 * 	        screen). A new image is generated when you press a key (except viewing the description of the fractal,
 * 	        and changing the zoom increment).
 *
 * 	        *****MAKE SURE NUM LOCK IS UNLOCKED OR THE PROGRAM WON'T WORK!!!*****
 *
 * How to run: make cudaJulia
 * 	       (make sure there isn't another cudaJulia executable in the directory.
 * 	       run cudaJulia
 * 
 */


#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <sys/time.h>


using namespace std;
using namespace cv;

int main(int argc, char** argv){
   
  //on each iteration, program calculation new = (old*old) + c
  //c is a constant
  //old starts at current pixel
  
  //double pr, pi; //real and imaginary part of pixel 'p'
  double cRe, cIm; //real and imaginary part of constant c
		   //determines shape of fractal
  double newRe, newIm, oldRe, oldIm; //real and imaginary parts of new and old
  double zoom = 1; //change the following to zoom and change position
  double zoomInc = 1;
  double moveX = 0.5;
  double moveY = 0;
  //color representation here
  int maxIterations = 128; //how many iterations should the function stop

  int imgXSize;
  int imgYSize;
  int localKey;
  bool waitForUserInput;
  bool done = false;

  struct timeval startTime, stopTime;
  double start, stop; //Variables to report time values

  IplImage* pic, *pic2;
  uchar *data;

  imgXSize = 1216; //xsize
  imgYSize = 1024; //ysize
  cRe = -0.7; //determines shape of julia set
  cIm = 0.27015;

  pic = cvCreateImage( //create a multi-channel byte image
		      cvSize(imgXSize, imgYSize), 
		      IPL_DEPTH_8U, 3);
  pic2 = cvCreateImage(
		      cvSize(imgXSize, imgYSize), 
		      IPL_DEPTH_8U, 3);

  cvNamedWindow("Julia fractal", CV_WINDOW_AUTOSIZE); 	//window for the image

  //int localAccess;
  cout << pic->height << endl;
  int height = pic->height;//1000
  int width = pic->width;
  int step = pic->widthStep/sizeof(uchar);
  int channels = pic->nChannels;
  data = (uchar *)pic->imageData;

while(!done)
{
   gettimeofday(&startTime, NULL);

   waitForUserInput = true;
   cout << string(50, '\n');
   cout << "Arrows move X,Y position" << endl;
   cout << "Keypad +,- zooms in and out" << endl;
   cout << "Keypad *,/ changes fractal iterations" << endl;
   cout << "Z and X keys changes real shape of fractal" << endl;
   cout << "C and V keys changes imaginary shape of fractal" << endl;
   cout << "A keys prompts user to change zoom increment" << endl;
   cout << endl;


  imgXSize = imgYSize = 0;

  for(int y = 0; y < height; y++){ //draw the fractal, output this for loop iterate 1000 times

     for(int x = 0; x < width; x++){
	
	//calculate the initial real and imaginary part of z
	//based on the pixel location, zoom, and position values
	
	newRe = 1.5 * ((x - (width))/2) / (0.5 * zoom * (width)) + moveX;
	newIm = (y - ( (height)/2 ) ) / (0.5 * zoom * (height)) + moveY; 

        //newRe = newIm = oldRe = oldIm = 0;

	int i; //the number of iterations

	for(i = 0; i < maxIterations; i++)
	{
	   oldRe = newRe; //remember previous iteration
	   oldIm = newIm;

	   newRe = (oldRe * oldRe) - (oldIm * oldIm) + cRe;
	   newIm = 2 * oldRe * oldIm + cIm;

	   //if the point is outside the circle w/ radius 2, then stop
	   if((newRe * newRe + newIm * newIm) > 4) break;
	}


	data[(imgYSize*step) + (imgXSize*channels + 0)] = i % 256;
	data[(imgYSize*step) + (imgXSize*channels + 1)] = 255; 
	data[(imgYSize*step) + (imgXSize*channels + 2)] = 255 * (i < maxIterations);


	imgXSize++;
     }
     cvCvtColor(pic, pic2, CV_HSV2BGR);

     imgXSize = 0;
     imgYSize++;

     cout << (char)0x0d << "Current line: " << imgYSize << flush;
     
   }
 
  cvCvtColor(pic, pic2, CV_HSV2BGR); //change the color scheme of the image
  cvShowImage("Julia fractal", pic2);
  gettimeofday(&stopTime, NULL);

   start = startTime.tv_sec + (startTime.tv_usec/1000000.0);
   stop = stopTime.tv_sec + (stopTime.tv_usec/1000000.0);

   printf("Image was generated in %f seconds\n", stop - start); //How long it took to generate image

   while(waitForUserInput)
   {
      localKey = waitKey(0); //wait indefinitely for user key press

   switch(localKey)
   {
      case 113: //Q, set parameters manually
	 cout << "X coord: ";
	 cin >> moveX;
	 cout << endl << "Y coord: ";
	 cin >> moveY;
	 cout << endl << "cRe: ";
	 cin >> cRe;
	 cout << endl << "cIm: ";
	 cin >> cIm;
	 cout << endl << "zoom: ";
	 cin >> zoom;
	 cout << endl << "max iterations: ";
	 cin >> maxIterations;
         cout << endl;
	 waitForUserInput = false;
	 break;
  
      case 65362: //up
	 moveY -= 0.25 * (1.0/zoom);	
	 waitForUserInput = false;
	 break;

      case 65364: //down
	 moveY += 0.25 * (1.0/zoom);
	 waitForUserInput = false;
	 break;

      case 65361: //left
	 moveX -= 0.25 * (1.0/zoom);
	 waitForUserInput = false;
	 break;

      case 65363: //right
	 moveX += 0.25 * (1.0/zoom);
	 waitForUserInput = false;
	 break;

      case 65451: //plus
	 zoom *= pow(1.001, zoomInc);
	 waitForUserInput = false;
	 break;

      case 65453: //minus
	 zoom /= pow(1.001, zoomInc);
	  waitForUserInput = false;
	 break;

      case 65450: //multiply
	 maxIterations *= 2;
	 waitForUserInput = false;
	 break;

      case 65455: //divide
	 if(maxIterations > 2)
	 {
	    maxIterations /= 2;
	    waitForUserInput = false;
	 }
	 else
	 {
	    cout << "Cannot zoom out any further" << endl;
	    waitForUserInput = true;
	 }
	 break;

      case 122: //Z key; Real part shape up
	 cRe += 0.075 * (1.0/zoom);
	 waitForUserInput = false;
	 break;

      case 120: //X key; REAL part shape down
	 cRe -= 0.075 * (1.0/zoom);
	 waitForUserInput = false;
	 break;

      case 99: //c key; Im part shape up
	 cIm += .075 * (1.0/zoom);
	 waitForUserInput = false;
	 break;

      case 118: //v key; Im part shape down
	 cIm -= .075 * (1.0/zoom);
	 waitForUserInput = false;
	 break;

      case 97: //a key; cin asks for new zoom increment
	 cout << endl << "Set zoom increment (must be greater than 0): ";
	 cin >> zoomInc;
	 cout << "zoomIncrement successfully set" << endl;
	 waitForUserInput = true;
	 break;

      case 100: //d key; description of coordinates
	 cout << endl << endl << "X Coord: " << moveX;
	 cout << endl << "Y Coord: " << moveY << endl;
	 cout << "cRe: " << cRe << endl;
	 cout << "cIm: " << cIm << endl;
	 cout << "Zoom: " << zoom << endl;
	 cout << "Zoom Increment: " << zoomInc << endl;
	 cout << "Max Iterations: " << maxIterations << endl;
	 waitForUserInput = true;
	 break;

      case 27: //esc key; exit
	 done = true;
	 waitForUserInput = false;
	 break;

   } //end switch
   } //end wait for user input

} //end done

// release the image
//



cvReleaseImage(&pic );


return 0;

} //end main
