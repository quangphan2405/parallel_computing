#include "parallel.h"


__kernel void physicsEngineKernel(__global satelite* satelites) {

    // Get current satellite ID
    size_t globalId = get_global_id(0);

    // double precision required for accumulation inside this routine,
    // but float storage is ok outside these loops.	
    __private doublevector tmpPosition;
	__private doublevector tmpVelocity;
	tmpPosition.x = satelites[globalId].position.x;
    tmpPosition.y = satelites[globalId].position.y;
    tmpVelocity.x = satelites[globalId].velocity.x;
    tmpVelocity.y = satelites[globalId].velocity.y;   

    // Physics iteration loop
    for(int physicsUpdateIndex = 0; 
        physicsUpdateIndex < PHYSICSUPDATESPERFRAME;
      ++physicsUpdateIndex){

        // Distance to the blackhole (bit ugly code because C-struct cannot have member functions)
        doublevector positionToBlackHole = {.x = tmpPosition.x -
            HORIZONTAL_CENTER, .y = tmpPosition.y - VERTICAL_CENTER};
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
        tmpVelocity.x -= accumulation * normalizedDirection.x *
            DELTATIME / PHYSICSUPDATESPERFRAME;
        tmpVelocity.y -= accumulation * normalizedDirection.y *
            DELTATIME / PHYSICSUPDATESPERFRAME;

        // Update position based on velocity
        tmpPosition.x +=
            tmpVelocity.x * DELTATIME / PHYSICSUPDATESPERFRAME;
        tmpPosition.y +=
            tmpVelocity.y * DELTATIME / PHYSICSUPDATESPERFRAME;

    }   

    // double precision required for accumulation inside this routine,
    // but float storage is ok outside these loops.
    // copy back the float storage.
    satelites[globalId].position.x = tmpPosition.x;
    satelites[globalId].position.y = tmpPosition.y;
    satelites[globalId].velocity.x = tmpVelocity.x;
    satelites[globalId].velocity.y = tmpVelocity.y;

}


__kernel void graphicsEngineKernel(__global satelite* satelites,
                                   __global color* pixels) {
	
    // Get global x,y coordinates
    size_t globalId_x = get_global_id(1);
    size_t globalId_y = get_global_id(0);

    // Row wise ordering
    __private floatvector pixel = { .x = globalId_x, .y = globalId_y};

    // This color is used for coloring the pixel
    __private color renderColor = { .red = 0.f, .green = 0.f, .blue = 0.f };
    __private color incrementColor = { .red = 0.f, .green = 0.f, .blue = 0.f };

    // Find closest satelite
    float shortestDistance = INFINITY;

    float weights = 0.f;
    int hitsSatellite = 0;

    // Graphics satelite loop: Find the closest satellite.
    for (int j = 0; j < SATELITE_COUNT; ++j) {

        floatvector difference = { .x = pixel.x - satelites[j].position.x,
                                    .y = pixel.y - satelites[j].position.y };
        float distance = sqrt(difference.x * difference.x +
                              difference.y * difference.y);

        
        float weight = 1.0f / (distance * distance * distance * distance);
        weights += weight;            
        if (distance < shortestDistance) {
            shortestDistance = distance;
            renderColor = satelites[j].identifier;
        }            
        incrementColor.red += satelites[j].identifier.red * weight;
        incrementColor.green += satelites[j].identifier.green * weight;
        incrementColor.blue += satelites[j].identifier.blue * weight;
        
    }

    // Calculate the color based on distance to every satelite.
    if (shortestDistance < SATELITE_RADIUS) {

        renderColor.red = 1.0f;
        renderColor.green = 1.0f;
        renderColor.blue = 1.0f;        

    } else {
        
        renderColor.red   += incrementColor.red / weights * 3.0f;                                                      
        renderColor.green += incrementColor.green / weights * 3.0f;                             	 
        renderColor.blue  += incrementColor.blue / weights * 3.0f;

    }

    pixels[globalId_x + WINDOW_WIDTH * globalId_y] = renderColor;
   
} 
