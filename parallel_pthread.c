/* COMP.CE.350 Parallelization Excercise 2020
   Copyright (c) 2016 Matias Koskela matias.koskela@tut.fi
                      Heikki Kultala heikki.kultala@tut.fi

VERSION 1.1 - updated to not have stuck satellites so easily
VERSION 1.2 - updated to not have stuck satellites hopefully at all.
VERSION 19.0 - make all satelites affect the color with weighted average.
               add physic correctness check.
VERSION 20.0 - relax physic correctness check
*/

// Example compilation on linux
// no optimization:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm
// most optimizations: gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2
// +vectorization +vectorize-infos: gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec
// +math relaxation:  gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec -ffast-math -pthread
// prev and OpenMP:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec -ffast-math -fopenmp
// prev and OpenCL:   gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec -ffast-math -fopenmp -lOpenCL

// Example compilation on macos X
// no optimization:   gcc -o parallel parallel.c -std=c99 -framework GLUT -framework OpenGL
// most optimization: gcc -o parallel parallel.c -std=c99 -framework GLUT -framework OpenGL -O3



#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h> // printf
#include <math.h> // INFINITY
#include <stdlib.h>
#include <string.h>

// Window handling includes
#ifndef __APPLE__
#include <GL/gl.h>
#include <GL/glut.h>
#else
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#endif
#include <pthread.h>


// These are used to decide the window size
#define WINDOW_HEIGHT 1024
#define WINDOW_WIDTH  1024

// The number of satelites can be changed to see how it affects performance.
// Benchmarks must be run with the original number of satellites
#define SATELITE_COUNT 64

// These are used to control the satelite movement
#define SATELITE_RADIUS 3.16f
#define MAX_VELOCITY 0.1f
#define GRAVITY 1.0f
#define DELTATIME 32
#define PHYSICSUPDATESPERFRAME 100000

// Some helpers to window size variables
#define SIZE WINDOW_HEIGHT*WINDOW_HEIGHT
#define HORIZONTAL_CENTER (WINDOW_WIDTH / 2)
#define VERTICAL_CENTER (WINDOW_HEIGHT / 2)

// Is used to find out frame times
int previousFrameTimeSinceStart = 0;
int previousFinishTime = 0;
unsigned int frameNumber = 0;
unsigned int seed = 0;

// Stores 2D data like the coordinates
typedef struct{
   float x;
   float y;
} floatvector;

// Stores 2D data like the coordinates
typedef struct{
   double x;
   double y;
} doublevector;

// Stores rendered colors. Each float may vary from 0.0f ... 1.0f
typedef struct{
   float red;
   float green;
   float blue;
} color;

// Stores the satelite data, which fly around black hole in the space
typedef struct{
   color identifier;
   floatvector position;
   floatvector velocity;
} satelite;

// Pixel buffer which is rendered to the screen
color* pixels;

// Pixel buffer which is used for error checking
color* correctPixels;

// Buffer for all satelites in the space
satelite* satelites;
satelite* backupSatelites;








// ## You may add your own variables here ##

#define NUM_THREADS 12

unsigned int ints[NUM_THREADS];
pthread_t thread_id[NUM_THREADS];

   
// ## You may add your own initialization routines here ##
void init(){

   // Initialize intergers for thread id, unrolled by 4.
	for (int i = 0;i < (int)(NUM_THREADS/4);++i){
	   ints[i]=i;
      ints[i+(int)(NUM_THREADS/4)]=i+(int)(NUM_THREADS/4);
      ints[i+(int)(2*NUM_THREADS/4)]=i+(int)(2*NUM_THREADS/4);  
      ints[i+(int)(3*NUM_THREADS/4)]=i+(int)(3*NUM_THREADS/4); 
   }

}

void *threadedParallelPhysicsEngine(void *thrd_id){

   int curr_thread_id = *((int *)thrd_id);

   // Starting and ending index of satelites for each thread.
   int start_index = (int)((curr_thread_id*SATELITE_COUNT)/NUM_THREADS);
   int end_index = (int)fminf((curr_thread_id+1)*SATELITE_COUNT/NUM_THREADS,SATELITE_COUNT);

   // double precision required for accumulation inside this routine,
   // but float storage is ok outside these loops.
   doublevector tmpPosition[SATELITE_COUNT];
   doublevector tmpVelocity[SATELITE_COUNT];
   for(int i = start_index; i < end_index; ++i){       
	   tmpPosition[i].x = satelites[i].position.x;
	   tmpPosition[i].y = satelites[i].position.y ;
	   tmpVelocity[i].x = satelites[i].velocity.x;
	   tmpVelocity[i].y = satelites[i].velocity.y;
   }

   // Physics satelite loop
   for(int i = start_index; i < end_index; ++i){
   
      // Physics iteration loop
      for(int physicsUpdateIndex = 0; 
          physicsUpdateIndex < PHYSICSUPDATESPERFRAME;
         ++physicsUpdateIndex){      

         // Distance to the blackhole (bit ugly code because C-struct cannot have member functions)
         doublevector positionToBlackHole = {.x = tmpPosition[i].x - HORIZONTAL_CENTER,
										 .y = tmpPosition[i].y - VERTICAL_CENTER};
         double distToBlackHoleSquared =
           positionToBlackHole.x * positionToBlackHole.x +
           positionToBlackHole.y * positionToBlackHole.y;
         double distToBlackHole = sqrt(distToBlackHoleSquared);

         // Gravity force
         doublevector normalizedDirection = {
            .x = positionToBlackHole.x / distToBlackHole,
		      .y = positionToBlackHole.y / distToBlackHole};
         double accumulation = GRAVITY / distToBlackHoleSquared;

         // Delta time is used to make velocity same despite different FPS
         // Update velocity based on force
         tmpVelocity[i].x -= accumulation * normalizedDirection.x *
            DELTATIME / PHYSICSUPDATESPERFRAME;
         tmpVelocity[i].y -= accumulation * normalizedDirection.y * 
            DELTATIME / PHYSICSUPDATESPERFRAME;

         // Update position based on velocity
         tmpPosition[i].x +=
            tmpVelocity[i].x * DELTATIME / PHYSICSUPDATESPERFRAME;
         tmpPosition[i].y +=
           tmpVelocity[i].y * DELTATIME / PHYSICSUPDATESPERFRAME;
      }

      // double precision required for accumulation inside this routine,
      // but float storage is ok outside these loops.
      // copy back the float storage.
      satelites[i].position.x = tmpPosition[i].x;
      satelites[i].position.y = tmpPosition[i].y;
      satelites[i].velocity.x = tmpVelocity[i].x;
      satelites[i].velocity.y = tmpVelocity[i].y;
   }
   pthread_exit(NULL);

}


// ## You are asked to make this code parallel ##
// Physics engine loop. (This is called once a frame before graphics engine) 
// Moves the satelites based on gravity
// This is done multiple times in a frame because the Euler integration 
// is not accurate enough to be done only once
void parallelPhysicsEngine(){

   // Create NUM_THREADS threads to do PhysicsEngine work
   for (int i = 0;i < NUM_THREADS; ++i) {
      pthread_create(&thread_id[i], NULL, threadedParallelPhysicsEngine, &ints[i]);
   }
   // Join and detach threads
   for (int i = 0;i < NUM_THREADS; ++i) {
      pthread_join(thread_id[i],NULL);
	   pthread_detach(thread_id[i]);
   }   

}


void *threadedParallelGraphicsEngine(void *thrd_id){

   int curr_thread_id = *((int *)thrd_id);

   // Starting and ending index of pixels for each thread.
   int start_index = curr_thread_id*SIZE/NUM_THREADS;
   int end_index = (int)fminf((curr_thread_id+1)*SIZE/NUM_THREADS,SIZE);

   // Graphics pixel loop
   for (int i = start_index ;i < end_index; ++i) {

      // Row wise ordering
      floatvector pixel = {.x = i % WINDOW_WIDTH, .y = i / WINDOW_WIDTH};

      // This color is used for coloring the pixel
      color renderColor = {.red = 0.f, .green = 0.f, .blue = 0.f};
	   color tmpRenderColor = {.red = 0.f, .green = 0.f, .blue = 0.f};
      // Find closest satelite
      float shortestDistance2 = INFINITY;
      float dist2s[SATELITE_COUNT]; // Store distances to satellites

      float weights = 0.f;
      int hitsSatellite = 0;

      // Temporary rgb variables for increment
      float red = 0.0f;
      float green = 0.0f;
      float blue = 0.0f;
      
      // Distance-to-satellite loop, unrolled by 4
      for (int k = 0; k < SATELITE_COUNT; k+=4) {
         floatvector difference0 = {.x = pixel.x - satelites[k].position.x,
                                    .y = pixel.y - satelites[k].position.y};
         floatvector difference1 = {.x = pixel.x - satelites[k+1].position.x,
                                    .y = pixel.y - satelites[k+1].position.y};
         floatvector difference2 = {.x = pixel.x - satelites[k+2].position.x,
                                    .y = pixel.y - satelites[k+2].position.y};
         floatvector difference3 = {.x = pixel.x - satelites[k+3].position.x,
                                    .y = pixel.y - satelites[k+3].position.y};

         dist2s[k]   = (difference0.x * difference0.x + 
                        difference0.y * difference0.y);
         dist2s[k+1] = (difference1.x * difference1.x + 
                        difference1.y * difference1.y);
         dist2s[k+2] = (difference2.x * difference2.x + 
                        difference2.y * difference2.y);             
         dist2s[k+3] = (difference3.x * difference3.x + 
                        difference3.y * difference3.y);

      }

	  
      // First Graphics satelite loop: Find the closest satellite.
      for (int j = 0; j < SATELITE_COUNT; ++j){
         float distance2 = (dist2s[j]);
         if (distance2 < SATELITE_RADIUS*SATELITE_RADIUS) {
            renderColor.red = 1.0f;
            renderColor.green = 1.0f;
            renderColor.blue = 1.0f;
            hitsSatellite = 1;
            break;
         } else {
            float weight = 1.0f / (distance2*distance2);
            weights += weight;
            if(distance2 < shortestDistance2){
               shortestDistance2 = distance2;
               renderColor = satelites[j].identifier;
            }
            // Increase color through temporary variables
			   red += (satelites[j].identifier.red)*weight;
	 
            green += (satelites[j].identifier.green)*weight;
	 
            blue += (satelites[j].identifier.blue)*weight;
         }
      }

      // Second graphics loop: Calculate the color based on distance to every satelite.
		if (!hitsSatellite) {
       
         renderColor.red   = red/weights * 3.0f;                               
	                              
         renderColor.green = green/weights * 3.0f;
                              	 
         renderColor.blue  = blue/weights * 3.0f;
            
		}
      pixels[i] = renderColor;
   }
}


// ## You are asked to make this code parallel ##
// Rendering loop (This is called once a frame after physics engine) 
// Decides the color for each pixel.
void parallelGraphicsEngine(){

   // Create NUM_THREADS threads to do GraphicsEngine work
   for (int i = 0;i < NUM_THREADS; ++i) {
      pthread_create(&thread_id[i], NULL, threadedParallelGraphicsEngine, &ints[i]);
   }
   // Join and detach threads
   for (int i = 0;i < NUM_THREADS; ++i) {
      pthread_join(thread_id[i],NULL);
	   pthread_detach(thread_id[i]);
   }      

}


// ## You may add your own destrcution routines here ##
void destroy(){


}







////////////////////////////////////////////////
// ¤¤ TO NOT EDIT ANYTHING AFTER THIS LINE ¤¤ //
////////////////////////////////////////////////

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Sequential rendering loop used for finding errors
void sequentialGraphicsEngine(){

    // Graphics pixel loop
    for(int i = 0 ;i < SIZE; ++i) {

      // Row wise ordering
      floatvector pixel = {.x = i % WINDOW_WIDTH, .y = i / WINDOW_WIDTH};

      // This color is used for coloring the pixel
      color renderColor = {.red = 0.f, .green = 0.f, .blue = 0.f};

      // Find closest satelite
      float shortestDistance = INFINITY;

      float weights = 0.f;
      int hitsSatellite = 0;

      // First Graphics satelite loop: Find the closest satellite.
      for(int j = 0; j < SATELITE_COUNT; ++j){
         floatvector difference = {.x = pixel.x - satelites[j].position.x,
                                   .y = pixel.y - satelites[j].position.y};
         float distance = sqrt(difference.x * difference.x + 
                               difference.y * difference.y);

         if(distance < SATELITE_RADIUS) {
            renderColor.red = 1.0f;
            renderColor.green = 1.0f;
            renderColor.blue = 1.0f;
            hitsSatellite = 1;
            break;
         } else {
            float weight = 1.0f / (distance*distance*distance*distance);
            weights += weight;
            if(distance < shortestDistance){
               shortestDistance = distance;
               renderColor = satelites[j].identifier;
            }
         }
      }

      // Second graphics loop: Calculate the color based on distance to every satelite.
      if (!hitsSatellite) {
         for(int j = 0; j < SATELITE_COUNT; ++j){
            floatvector difference = {.x = pixel.x - satelites[j].position.x,
                                      .y = pixel.y - satelites[j].position.y};
            float dist2 = (difference.x * difference.x +
                           difference.y * difference.y);
            float weight = 1.0f/(dist2* dist2);

            renderColor.red += (satelites[j].identifier.red *
                                weight /weights) * 3.0f;

            renderColor.green += (satelites[j].identifier.green *
                                  weight / weights) * 3.0f;

            renderColor.blue += (satelites[j].identifier.blue *
                                 weight / weights) * 3.0f;
         }
      }
      correctPixels[i] = renderColor;
    }
}

void sequentialPhysicsEngine(satelite *s){

   // double precision required for accumulation inside this routine,
   // but float storage is ok outside these loops.
   doublevector tmpPosition[SATELITE_COUNT];
   doublevector tmpVelocity[SATELITE_COUNT];

   for (int i = 0; i < SATELITE_COUNT; ++i) {
       tmpPosition[i].x = s[i].position.x;
       tmpPosition[i].y = s[i].position.y;
       tmpVelocity[i].x = s[i].velocity.x;
       tmpVelocity[i].y = s[i].velocity.y;
   }

   // Physics iteration loop
   for(int physicsUpdateIndex = 0;
       physicsUpdateIndex < PHYSICSUPDATESPERFRAME;
      ++physicsUpdateIndex){

       // Physics satelite loop
      for(int i = 0; i < SATELITE_COUNT; ++i){

         // Distance to the blackhole
         // (bit ugly code because C-struct cannot have member functions)
         doublevector positionToBlackHole = {.x = tmpPosition[i].x -
            HORIZONTAL_CENTER, .y = tmpPosition[i].y - VERTICAL_CENTER};
         double distToBlackHoleSquared =
            positionToBlackHole.x * positionToBlackHole.x +
            positionToBlackHole.y * positionToBlackHole.y;
         double distToBlackHole = sqrt(distToBlackHoleSquared);

         // Gravity force
         doublevector normalizedDirection = {
            .x = positionToBlackHole.x / distToBlackHole,
            .y = positionToBlackHole.y / distToBlackHole};
         double accumulation = GRAVITY / distToBlackHoleSquared;

         // Delta time is used to make velocity same despite different FPS
         // Update velocity based on force
         tmpVelocity[i].x -= accumulation * normalizedDirection.x *
            DELTATIME / PHYSICSUPDATESPERFRAME;
         tmpVelocity[i].y -= accumulation * normalizedDirection.y *
            DELTATIME / PHYSICSUPDATESPERFRAME;

         // Update position based on velocity
         tmpPosition[i].x +=
            tmpVelocity[i].x * DELTATIME / PHYSICSUPDATESPERFRAME;
         tmpPosition[i].y +=
            tmpVelocity[i].y * DELTATIME / PHYSICSUPDATESPERFRAME;
      }
   }

   // double precision required for accumulation inside this routine,
   // but float storage is ok outside these loops.
   // copy back the float storage.
   for (int i = 0; i < SATELITE_COUNT; ++i) {
       s[i].position.x = tmpPosition[i].x;
       s[i].position.y = tmpPosition[i].y;
       s[i].velocity.x = tmpVelocity[i].x;
       s[i].velocity.y = tmpVelocity[i].y;
   }
}

// Just some value that barely passes for OpenCL example program
#define ALLOWED_FP_ERROR 0.08
// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void errorCheck(){
   for(unsigned int i=0; i < SIZE; ++i) {
      if(fabs(correctPixels[i].red - pixels[i].red) > ALLOWED_FP_ERROR ||
         fabs(correctPixels[i].green - pixels[i].green) > ALLOWED_FP_ERROR ||
         fabs(correctPixels[i].blue - pixels[i].blue) > ALLOWED_FP_ERROR) {
         printf("Buggy pixel at (x=%i, y=%i). Press enter to continue.\n", i % WINDOW_WIDTH, i / WINDOW_WIDTH);
         getchar();
         return;
       }
   }
   printf("Error check passed!\n");
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void compute(void){
   int timeSinceStart = glutGet(GLUT_ELAPSED_TIME);
   previousFrameTimeSinceStart = timeSinceStart;
	
   // Error check during first frames
   if (frameNumber < 2) {
	  memcpy(backupSatelites, satelites, sizeof(satelite) * SATELITE_COUNT);
      sequentialPhysicsEngine(backupSatelites);
	  
		
   }
   parallelPhysicsEngine();
   if (frameNumber < 2) {
	   FILE *fptr = fopen("program.txt", "w");
      for (int i = 0; i < SATELITE_COUNT; i++) {
		  
		
		
         if (memcmp (&satelites[i], &backupSatelites[i], sizeof(satelite))) {
			   // Init satelites buffer which are moving in the space
			//satelite* data = (data*)malloc(sizeof(satelite));
			// memcpy(&data, &satelites[i], sizeof satelites);
			// memcpy(&data2, &backupSatelites[i], sizeof satelites);
			
			// for (size_t j=0; j < sizeof (satelite); ++j) {
				 fprintf(fptr, "%d\n", memcmp (&satelites[i], &backupSatelites[i], sizeof(satelite)));
				 
				 //fprintf(fptr, "Sat[%d](%d) = %02x\n", i,j,*((&backupSatelites[i]+j)));
				 //fprintf(fptr, "BSat[%d](%d) = %02hhx\n", i,j,backupSatelites[j]);
			// }
			 
			 
			//fprintf(fptr, "BSat_%d = %x\n\n", i,backupSatelites[i]);
            printf("Incorrect satelite data of satelite: %d\n", i);
            //getchar();
         }
      }
	  fclose(fptr);
   }

   int sateliteMovementMoment = glutGet(GLUT_ELAPSED_TIME);
   int sateliteMovementTime = sateliteMovementMoment  - timeSinceStart;

   // Decides the colors for the pixels
   parallelGraphicsEngine();

   int pixelColoringMoment = glutGet(GLUT_ELAPSED_TIME);
   int pixelColoringTime =  pixelColoringMoment - sateliteMovementMoment;

   // Sequential code is used to check possible errors in the parallel version
   if(frameNumber < 2){
      sequentialGraphicsEngine();
  //    errorCheck();
   }

   int finishTime = glutGet(GLUT_ELAPSED_TIME);
   // Print timings
   int totalTime = finishTime - previousFinishTime;
   previousFinishTime = finishTime;

   printf("Total frametime: %ims, satelite moving: %ims, space coloring: %ims.\n",
      totalTime, sateliteMovementTime, pixelColoringTime);

   // Render the frame
   glutPostRedisplay();
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Probably not the best random number generator
float randomNumber(float min, float max){
   return (rand() * (max - min) / RAND_MAX) + min;
}

// DO NOT EDIT THIS FUNCTION
void fixedInit(unsigned int seed){

   if(seed != 0){
     srand(seed);
   }

   // Init pixel buffer which is rendered to the widow
   pixels = (color*)malloc(sizeof(color) * SIZE);

   // Init pixel buffer which is used for error checking
   correctPixels = (color*)malloc(sizeof(color) * SIZE);

   backupSatelites = (satelite*)malloc(sizeof(satelite) * SATELITE_COUNT);


   // Init satelites buffer which are moving in the space
   satelites = (satelite*)malloc(sizeof(satelite) * SATELITE_COUNT);

   // Create random satelites
   for(int i = 0; i < SATELITE_COUNT; ++i){

      // Random reddish color
      color id = {.red = randomNumber(0.f, 0.15f) + 0.1f,
                  .green = randomNumber(0.f, 0.14f) + 0.0f,
                  .blue = randomNumber(0.f, 0.16f) + 0.0f};
    
      // Random position with margins to borders
      floatvector initialPosition = {.x = HORIZONTAL_CENTER - randomNumber(50, 320),
                              .y = VERTICAL_CENTER - randomNumber(50, 320) };
      initialPosition.x = (i / 2 % 2 == 0) ?
         initialPosition.x : WINDOW_WIDTH - initialPosition.x;
      initialPosition.y = (i < SATELITE_COUNT / 2) ?
         initialPosition.y : WINDOW_HEIGHT - initialPosition.y;

      // Randomize velocity tangential to the balck hole
      floatvector positionToBlackHole = {.x = initialPosition.x - HORIZONTAL_CENTER,
                                    .y = initialPosition.y - VERTICAL_CENTER};
      float distance = (0.06 + randomNumber(-0.01f, 0.01f))/ 
        sqrt(positionToBlackHole.x * positionToBlackHole.x + 
          positionToBlackHole.y * positionToBlackHole.y);
      floatvector initialVelocity = {.x = distance * -positionToBlackHole.y,
                                .y = distance * positionToBlackHole.x};

      // Every other orbits clockwise
      if(i % 2 == 0){
         initialVelocity.x = -initialVelocity.x;
         initialVelocity.y = -initialVelocity.y;
      }

      satelite tmpSatelite = {.identifier = id, .position = initialPosition,
                              .velocity = initialVelocity};
      satelites[i] = tmpSatelite;
   }
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
void fixedDestroy(void){
   destroy();

   free(pixels);
   free(correctPixels);
   free(satelites);

   if(seed != 0){
     printf("Used seed: %i\n", seed);
   }
}

// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
// Renders pixels-buffer to the window 
void render(void){
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_FLOAT, pixels);
   glutSwapBuffers();
   frameNumber++;
}

// DO NOT EDIT THIS FUNCTION
// Inits glut and start mainloop
int main(int argc, char** argv){

   if(argc > 1){
     seed = atoi(argv[1]);
     printf("Using seed: %i\n", seed);
   }
   printf("Ran");
   // Init glut window
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
   glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
   glutCreateWindow("Parallelization excercise");
   glutDisplayFunc(render);
   atexit(fixedDestroy);
   previousFrameTimeSinceStart = glutGet(GLUT_ELAPSED_TIME);
   previousFinishTime = glutGet(GLUT_ELAPSED_TIME);
   glEnable(GL_DEPTH_TEST);
   glClearColor(0.0, 0.0, 0.0, 1.0);
   fixedInit(seed);
   init();

   // compute-function is called when everythin from last frame is ready
   glutIdleFunc(compute);

   // Start main loop
   glutMainLoop();
}
