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
// +math relaxation:  gcc -o parallel parallel.c -std=c99 -lglut -lGL -lm -O2 -ftree-vectorize -fopt-info-vec -ffast-math
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

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/opencl.h> // OpenCL
#include <assert.h> // assert
#include "parallel.h" // Header file

// Window handling includes
#ifndef __APPLE__
#include <GL/gl.h>
#include <GL/glut.h>
#else
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#endif


// Is used to find out frame times
int previousFrameTimeSinceStart = 0;
int previousFinishTime = 0;
unsigned int frameNumber = 0;
unsigned int seed = 0;


// Pixel buffer which is rendered to the screen
color* pixels;

// Pixel buffer which is used for error checking
color* correctPixels;

// Buffer for all satelites in the space
satelite* satelites;
satelite* backupSatelites;


#define TOTAL_PIXEL_SIZE sizeof(color) * SIZE
#define TOTAL_SATELLITE_SIZE sizeof(satelite) * SATELITE_COUNT
#define MAX_SOURCE_SIZE (0x100000)



// ## You may add your own variables here ##
cl_command_queue physicsCommandQueue = NULL;
cl_command_queue graphicsCommandQueue = NULL;

cl_context physicsContext = NULL;
cl_context graphicsContext = NULL;

cl_mem physicsSatelitesBuffer = NULL;
cl_mem graphicsSatelitesBuffer = NULL;
cl_mem pixelsBuffer = NULL;

cl_program physicsProgram = NULL;
cl_program graphicsProgram = NULL;

cl_kernel physicsKernel = NULL;
cl_kernel graphicsKernel = NULL;

cl_int err;
cl_event k_events;

size_t local_size[2];

const char* option = "-I parallel.h -cl-fast-relaxed-math"; 




// Take workgroup size from user
void setLocalSize() {

   printf("Enter the WG x-size: ");
   assert(scanf("%zu", &local_size[0]) > 0);
   printf("Enter the WG y-size: ");
   assert(scanf("%zu", &local_size[1]) > 0);

}


// Get the ID of the desired device type.
cl_device_id getDeviceID(cl_device_type device_type) {

    cl_platform_id *platforms;
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint numPlatforms;   
    cl_uint numDevices;  
   
    // Get total number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    assert(err == CL_SUCCESS);
   
    // Get the platforms using malloc
    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    assert(err == CL_SUCCESS);
   
    // Get the first device in the first available platform
    for (int i = 0; i < numPlatforms; ++i) {
        err = clGetDeviceIDs(platforms[i], device_type, 0, NULL, &numDevices);
        cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
        clGetDeviceIDs(platforms[i], device_type, numDevices, devices, NULL);

        if (err == CL_SUCCESS) {
            return devices[0];
        } else {
            continue;
        }
    }

} 


// Setup the CL properties for the Physics Engine
void setupPhysics(char* source_str, size_t source_size) {

    // CPU performs physics engine better
    cl_device_id cpuID = getDeviceID(CL_DEVICE_TYPE_CPU);

    // Create context for Physics Engine
    physicsContext = clCreateContext(NULL, 1, &cpuID, NULL, NULL, &err);
    assert(err == CL_SUCCESS);

    // Create command queue for Physics Engine
    physicsCommandQueue = clCreateCommandQueue(physicsContext, cpuID, 0, &err);
    assert(err == CL_SUCCESS);

    // Create buffer for satellites in Physics Engine
    physicsSatelitesBuffer = clCreateBuffer(physicsContext, CL_MEM_USE_HOST_PTR, 
                    TOTAL_SATELLITE_SIZE, satelites, &err);

    clFinish(physicsCommandQueue);

    // Create program from source string
    physicsProgram = clCreateProgramWithSource(physicsContext, 1, (const char **)&source_str,
                    (const size_t *)&source_size, &err);
    assert(err == CL_SUCCESS);

    // Build the program and print error if exist
    err = clBuildProgram(physicsProgram, 1, &cpuID, option, NULL, NULL);
    if (err != CL_SUCCESS) {
        char* buffErr;
        cl_int errCode;
        size_t logLen;

        errCode = clGetProgramBuildInfo(physicsProgram, cpuID,
                    CL_PROGRAM_BUILD_LOG, 0, NULL, &logLen);
        if (errCode) {
            printf("clGetProgramBuildInfo failed at line %d\n", errCode);
            exit(-1);
        }

        buffErr = malloc(logLen);
        if (!buffErr) {
            printf("Malloc failed at line %d\n", __LINE__);
            exit(-2);
        }

        errCode = clGetProgramBuildInfo(physicsProgram, cpuID,
                    CL_PROGRAM_BUILD_LOG, logLen, buffErr, NULL);
        if (errCode) {
            printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
            exit(-3);
        }

        fprintf(stderr, "Build log: \n%s\n", buffErr); //Be careful with  the fprint
        free(buffErr);
        fprintf(stderr, "clBuildProgram failed\n");
        exit(EXIT_FAILURE);
    }

    // Create Physics Engine kernel
    physicsKernel = clCreateKernel(physicsProgram, "physicsEngineKernel", &err);

    // Set argument for Physics Engine kernel
    err = clSetKernelArg(physicsKernel, 0, sizeof(cl_mem), (void*)&physicsSatelitesBuffer);
    assert(err == CL_SUCCESS);

    clFinish(physicsCommandQueue);

}


// Setup the CL properties for the Graphics Engine
void setupGraphics(char* source_str, size_t source_size) {

    // GPU performs graphics engine better
    cl_device_id gpuID = getDeviceID(CL_DEVICE_TYPE_GPU);

    // Create context for Graphics Engine
    graphicsContext = clCreateContext(NULL, 1, &gpuID, NULL, NULL, &err);
    assert(err == CL_SUCCESS);

    // Create command queue for Graphics Engine
    graphicsCommandQueue = clCreateCommandQueue(graphicsContext, gpuID, 0, &err);
    assert(err == CL_SUCCESS);

    // Create buffer for satellites and pixels in Graphics Engine
    graphicsSatelitesBuffer = clCreateBuffer(graphicsContext, CL_MEM_USE_HOST_PTR, 
                    TOTAL_SATELLITE_SIZE, satelites, &err);
    assert(err == CL_SUCCESS);

    pixelsBuffer = clCreateBuffer(graphicsContext, CL_MEM_USE_HOST_PTR, 
                    TOTAL_PIXEL_SIZE, pixels, &err); 
    assert(err == CL_SUCCESS); 

    clFinish(graphicsCommandQueue);

    // Create program from source string
    graphicsProgram = clCreateProgramWithSource(graphicsContext, 1, (const char **)&source_str,
                    (const size_t *)&source_size, &err);
    assert(err == CL_SUCCESS);

    // Build the program and print error if exist
    err = clBuildProgram(graphicsProgram, 1, &gpuID, option, NULL, NULL);
    if (err != CL_SUCCESS) {
        char* buffErr;
        cl_int errCode;
        size_t logLen;

        errCode = clGetProgramBuildInfo(graphicsProgram, gpuID,
                    CL_PROGRAM_BUILD_LOG, 0, NULL, &logLen);
        if (errCode) {
            printf("clGetProgramBuildInfo failed at line %d\n", errCode);
            exit(-1);
        }

        buffErr = malloc(logLen);
        if (!buffErr) {
            printf("Malloc failed at line %d\n", __LINE__);
            exit(-2);
        }

        errCode = clGetProgramBuildInfo(graphicsProgram, gpuID,
                    CL_PROGRAM_BUILD_LOG, logLen, buffErr, NULL);
        if (errCode) {
            printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
            exit(-3);
        }

        fprintf(stderr, "Build log: \n%s\n", buffErr); //Be careful with  the fprint
        free(buffErr);
        fprintf(stderr, "clBuildProgram failed\n");
        exit(EXIT_FAILURE);
    }   

    // Create Graphics Engine kernel
    graphicsKernel = clCreateKernel(graphicsProgram, "graphicsEngineKernel", &err);

    // Set arguments for Graphics Engine kernel
    err = clSetKernelArg(graphicsKernel, 0, sizeof(cl_mem), (void *)&graphicsSatelitesBuffer);
    err = clSetKernelArg(graphicsKernel, 1, sizeof(cl_mem), (void *)&pixelsBuffer);
    assert(err == CL_SUCCESS);

    clFinish(graphicsCommandQueue);

}


// ## You may add your own initialization routines here ##
void init(){

    // Read source code from file
    FILE *f;
    char *source_str;
    size_t source_size;

    f = fopen("parallel.cl", "r");
    if (!f) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, f);
    fclose(f);
   
    // Set up engines
    setupPhysics(source_str, source_size);
    setupGraphics(source_str, source_size);
    
    // Set workgroup size in Graphics Engine
    setLocalSize();

}


// ## You are asked to make this code parallel ##
// Physics engine loop. (This is called once a frame before graphics engine) 
// Moves the satelites based on gravity
// This is done multiple times in a frame because the Euler integration 
// is not accurate enough to be done only once
void parallelPhysicsEngine(){

    // Total number of satellites
    size_t global_size = SATELITE_COUNT;

    // Execute the Physics Engine kernel
    err = clEnqueueNDRangeKernel(physicsCommandQueue, physicsKernel, 
                1, NULL, &global_size, NULL, 0, NULL, NULL);

    // Wait for finishing in the first frames, after this 
    // Graphics Engine can take data simultaneously.
    if (frameNumber < 2) {
      clFinish(physicsCommandQueue);
    }

}


// ## You are asked to make this code parallel ##
// Rendering loop (This is called once a frame after physics engine) 
// Decides the color for each pixel.
void parallelGraphicsEngine(){

    // Wait for Physics Engine
    err = clWaitForEvents(1, &k_events);

    // Total number of pixels
    size_t global_size[2] = {WINDOW_HEIGHT, WINDOW_WIDTH};

    // Write satellite data to buffer
    err = clEnqueueWriteBuffer(graphicsCommandQueue, graphicsSatelitesBuffer,
                CL_TRUE, 0, TOTAL_SATELLITE_SIZE, satelites, 0, NULL, NULL);
    clFinish(graphicsCommandQueue);

    // Execute the Graphics Engine kernel
    err = clEnqueueNDRangeKernel(graphicsCommandQueue, graphicsKernel, 
                2, NULL, global_size, local_size, 0, NULL, NULL);
    clFinish(graphicsCommandQueue);

    // Read pixel data from buffer
    err = clEnqueueReadBuffer(graphicsCommandQueue, pixelsBuffer, CL_TRUE,
                0, TOTAL_PIXEL_SIZE, pixels, 0, NULL, NULL);
    clFlush(graphicsCommandQueue);
    clFinish(graphicsCommandQueue);

}


// ## You may add your own destrcution routines here ##
void destroy(){

    // Release OpenCL properties
    clReleaseMemObject(physicsSatelitesBuffer);
    clReleaseMemObject(graphicsSatelitesBuffer);
    clReleaseMemObject(pixelsBuffer);

    clReleaseCommandQueue(physicsCommandQueue);
    clReleaseCommandQueue(graphicsCommandQueue);

    clReleaseKernel(physicsKernel);
    clReleaseKernel(graphicsKernel);
   
    clReleaseProgram(physicsProgram);
    clReleaseProgram(graphicsProgram);
   
    clReleaseContext(physicsContext);
    clReleaseContext(graphicsContext);   

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
      for (int i = 0; i < SATELITE_COUNT; i++) {
         if (memcmp (&satelites[i], &backupSatelites[i], sizeof(satelite))) {
            printf("Incorrect satelite data of satelite: %d\n", i);
            getchar();
         }
      }
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
      errorCheck();
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