/*
 * Name: Aaron Marlowe
 * Date: 12/8/2016
 * Assignment: High Performance Final Project (PARALLEL CODE)
 * Description: This program generates a 1216x1024 image of a julia fractalusing CDA. This image can be manipulated
 *              (once loaded) by pressing a variety of keys to change the image parameters (instructions will be on
 *              screen). A new image is generated when you press a key (except viewing the description of the fractal,
 *              and changing the zoom increment).
 *
 *              *****MAKE SURE NUM LOCK IS UNLOCKED OR THE PROGRAM WON'T WORK!!!*****
 *
 * How to run: make testCuda
 *             (make sure there isn't another testCuda executable in the directory.
 *             run testCuda
 * Location: ~/HighPerformance/FinalProject/cv-fractals/trunk/C++ OpenCV/testCuda.cu
 */



#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <sys/time.h>

using namespace std;
using namespace cv;

typedef struct {
     double cRe, cIm; //real and imaginary parts of constant 'c'
     double newRe, newIm;
     double oldRe, oldIm;
     double zoom;
     double zoomInc;
     double moveX, moveY;
     int maxIterations; //Higher maxIterations equates to a more detailed fractal
     int imgXSize; //x size of image
     int imgYSize; // y size of image
     int nChannels; //number of channels in Image
     int step; //The number of elements we need to 'step' in order to get to another row in the image 
	       // i.e the number of x needed to pass by in order to increment y by 1
  } imageProperties;


void errorCheck(int code, cudaError_t err);
void preGPU(uchar *pic, imageProperties prop, int tile_width);
__global__ void calculateFractal(uchar *gpuData, double *d_data, int *i_data, int tile_width);
void clearScreen();


int main(int argc, char** argv){
  
  //on each iteration, program calculation new = (old*old) + c
  //c is a constant
  //old starts at current pixel

  int tile_width;
  int localKey;
  bool waitForUserInput;
  bool done = false;
  IplImage* pic, *pic2; //pointers to hold image data
  
  imageProperties prop;
  struct timeval startTime, stopTime;
  double start, stop; //Variables to report time values

  prop.zoom = 1.0;
  prop.zoomInc = 1.0;
  prop.moveX = 0.5;
  prop.moveY = 0.0;
  prop.maxIterations = 128;
  prop.imgXSize = 1216; //Choses x size of image
  prop.imgYSize = 1024; //Chosen y size of image
  prop.cRe = -0.7; //determine initial shape of julia set
  prop.cIm = 0.27015;

  tile_width = 32;

  pic = cvCreateImage( //create a multi-channel byte image
                      cvSize(prop.imgXSize, prop.imgYSize),
                      IPL_DEPTH_8U, 3);
  pic2 = cvCreateImage(
                      cvSize(prop.imgXSize, prop.imgYSize),
                      IPL_DEPTH_8U, 3);

  //width is first param, height is 2nd param of cvCreateImage


  int height = pic->height; //1024
  int width = pic->width; //1216
  prop.step = pic->widthStep/sizeof(uchar);
  prop.nChannels = pic->nChannels;

  cvNamedWindow("Julia fractal", CV_WINDOW_AUTOSIZE); 	


  while(!done)
  {
     gettimeofday(&startTime, NULL); //start timing our data
     uchar *data = new uchar[3 * prop.imgXSize * prop.imgYSize]; //will hold fractal data from gpu

     clearScreen();
     printf("Generating Image... \n");

     preGPU(data, prop, tile_width);

     for(int j = 0; j < width*height*3; j++ ) //iterate through every pixel of image
     {
	pic->imageData[j] = data[j]; //copy data from gpu to host
     }

     cvCvtColor(pic, pic2, CV_HSV2BGR); //pic is src, pic2 is dest


   cvShowImage("Julia fractal", pic2);
   gettimeofday(&stopTime, NULL);

   start = startTime.tv_sec + (startTime.tv_usec/1000000.0);
   stop = stopTime.tv_sec + (stopTime.tv_usec/1000000.0);

   printf("Image was generated in %f seconds\n", stop - start);

   waitForUserInput = true;

   printf("Arrows move X,Y position\n");
   printf("Keypad +,- zooms in and out\n");
   printf("Keypad *,/ changes fractal iterations\n");
   printf("Z and X keys changes real shape of fractal\n");
   printf("C and V keys changes imaginary shape of fractal\n");
   printf("A keys prompts user to change zoom increment\n\n");


   while(waitForUserInput)
   {
      printf("Waiting for User Input...\n");
      localKey = waitKey(0); //wait indefinitely for user key press

   switch(localKey)
   {
      case 65362: //up
	 prop.moveY -= 0.25 * (1.0/prop.zoom);	
	 waitForUserInput = false;
	 break;

      case 65364: //down
	 prop.moveY += 0.25 * (1.0/prop.zoom);
	 waitForUserInput = false;
	 break;

      case 65361: //left
	 prop.moveX -= 0.25 * (1.0/prop.zoom);
	 waitForUserInput = false;
	 break;

      case 65363: //right
	 prop.moveX += 0.25 * (1.0/prop.zoom);
	 waitForUserInput = false;
	 break;

      case 65451: //plus
	 prop.zoom *= pow(1.001, prop.zoomInc);
	 waitForUserInput = false;
	 break;

      case 65453: //minus
	 prop.zoom /= pow(1.001, prop.zoomInc);
	  waitForUserInput = false;
	 break;

      case 65450: //multiply
	 prop.maxIterations *= 2;
	 waitForUserInput = false;
	 break;

      case 65455: //divide
	 if(prop.maxIterations > 2)
	 {
	    prop.maxIterations /= 2;
	    waitForUserInput = false;
	 }
	 else
	 {
	    cout << "Cannot zoom out any further" << endl;
	    waitForUserInput = true;
	 }
	 break;

      case 122: //Z key; Real part shape up
	 prop.cRe += 0.075 * (1.0/prop.zoom);
	 waitForUserInput = false;
	 break;

      case 120: //X key; REAL part shape down
	 prop.cRe -= 0.075 * (1.0/prop.zoom);
	 waitForUserInput = false;
	 break;

      case 99: //c key; Im part shape up
	 prop.cIm += .075 * (1.0/prop.zoom);
	 waitForUserInput = false;
	 break;

      case 118: //v key; Im part shape down
	 prop.cIm -= .075 * (1.0/prop.zoom);
	 waitForUserInput = false;
	 break;

      case 97: //a key; cin asks for new zoom increment
	 cout << endl << "Set zoom increment (must be greater than 0): ";
	 cin >> prop.zoomInc;
	 cout << "zoomIncrement successfully set" << endl;
	 waitForUserInput = true;
	 break;

      case 100: //d key; description of coordinates
	 cout << endl << endl << "X Coord: " << prop.moveX;
	 cout << endl << "Y Coord: " << prop.moveY << endl;
	 cout << "cRe: " << prop.cRe << endl;
	 cout << "cIm: " << prop.cIm << endl;
	 cout << "Zoom: " << prop.zoom << endl;
	 cout << "Zoom Increment: " << prop.zoomInc << endl;
	 cout << "Max Iterations: " << prop.maxIterations << endl;
	 waitForUserInput = true;
	 break;

      case 27: //esc key; exit
	 done = true;
	 waitForUserInput = false;
	 break;

      default:
	 cout << endl << "Key Not Recognized" << endl;
 	 break;

   } //end switch
   } //end wait for user input

   delete[] data;

} //end done

// release the image
cvReleaseImage(&pic2);


cout << "Program completed" << endl;

return 0;

} //end main


void preGPU(uchar *pic, imageProperties prop, int tile_width)
{
   int gridSizeNumX;
   int gridSizeNumY;
   int byteCountData; //Total data amount to set aside on gpu
   double *dd_data;
   int *ii_data;
   uchar *gpuData; //holds the data on GPU side

   cudaSetDevice(0); 

   double d_data[10] = {prop.cRe, prop.cIm, prop.newRe, prop.newIm,
		   prop.oldRe, prop.oldIm, prop.zoom, prop.zoomInc,
		   prop.moveX, prop.moveY};

   
   int i_data[5] = {prop.maxIterations, prop.imgXSize, prop.imgYSize,
		    prop.nChannels, prop.step};

 
   gridSizeNumX = ceil(prop.imgXSize/(double)tile_width); //38 blocks in x dir
   gridSizeNumY = ceil(prop.imgYSize/(double)tile_width); //32 blocks in y dir


   dim3 numBlocks(gridSizeNumX, gridSizeNumY, 1);
   dim3 threadsPerBlock(tile_width, tile_width, 1);

   byteCountData = prop.imgXSize * (3 * prop.imgYSize) * sizeof(uchar);

   //Allocate memory on gpu side
   errorCheck(1, cudaMalloc((void**)&gpuData, byteCountData)); 
   errorCheck(3, cudaMalloc((void**)&dd_data, 10 * sizeof(double)));
   errorCheck(4, cudaMalloc((void**)&ii_data, 5 * sizeof(int)));
   errorCheck(5, cudaMemcpy(dd_data, d_data, 10 * sizeof(double), cudaMemcpyHostToDevice));
   errorCheck(6, cudaMemcpy(ii_data, i_data, 5  * sizeof(int), cudaMemcpyHostToDevice)); 


   //kernel call
   calculateFractal<<<numBlocks, threadsPerBlock>>>(gpuData, dd_data, ii_data, tile_width);

   cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",cudaGetErrorString(cudaerr));


   //get data back from kernel
   errorCheck(2, cudaMemcpy(pic, gpuData,  byteCountData, cudaMemcpyDeviceToHost));

   cudaFree(gpuData); //free GPU memory

} //end pre-kernel



__global__ void calculateFractal(uchar *gpuData, double *d_data, int *i_data, int tile_width)
{
   int x = blockIdx.x * tile_width + threadIdx.x;
   int y = blockIdx.y * tile_width + threadIdx.y;
  
   int height = i_data[2]; //imgYSize, 1024
   int width = i_data[1]; //imgXSize, 1216
   int maxIterations = i_data[0];
   int channels = i_data[3];
   int step = i_data[4];

   double cRe = d_data[0];
   double cIm = d_data[1];
   double newRe = d_data[2];
   double newIm = d_data[3];
   double oldRe = d_data[4];
   double oldIm = d_data[5];
   double zoom = d_data[6];
   double moveX = d_data[8];
   double moveY = d_data[9];

   int i; //the number of iterations

        //calculate the initial real and imaginary part of z
        //based on the pixel location, zoom, and position values

        newRe = 1.5 * ((x - (width))/2) / (0.5 * zoom * (width)) + moveX;
        newIm = (y - ( (height)/2 ) ) / (0.5 * zoom * (height)) + moveY;


        for(i = 0; i < maxIterations; i++)
        {
           oldRe = newRe; //remember previous iteration
           oldIm = newIm;

           newRe = (oldRe * oldRe) - (oldIm * oldIm) + cRe;
           newIm = 2 * oldRe * oldIm + cIm;

           //if the point is outside the circle w/ radius 2, then stop
           if((newRe * newRe + newIm * newIm) > 4) break;

        }
	
	//__syncthreads(); //all threads in a block synchronize

        gpuData[(y*step) + (x*channels + 0)] = i % 256;
        gpuData[(y*step) + (x*channels + 1)] = 255;
        gpuData[(y*step) + (x*channels + 2)] = 255 * (i < maxIterations);
  
} //end kernel

void errorCheck(int code, cudaError_t err)
{
   if(err != cudaSuccess){
      cout << "Code:  " << code << cudaGetErrorString(err) << "  in " 
	   << __FILE__ << " at line " << __LINE__ << endl;
      exit(EXIT_FAILURE);
   }
}


void clearScreen() //must be a better way of doing this
{
   for(int i = 0; i < 50; i++)
      printf("\n");
}





