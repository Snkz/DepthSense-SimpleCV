/*
 * DepthSense SDK for Python and SimpleCV
 * -----------------------------------------------------------------------------
 * file:            imageproccessing.cxx
 * author:          Abdi Dahir
 * modified:        May 9 2014
 * vim:             set fenc=utf-8:ts=4:sw=4:expandtab:
 * 
 * Convoloutions and other assorted proccessing techniques defined here
 * -----------------------------------------------------------------------------
 */

// MS completly untested
#ifdef _MSC_VER
#include <windows.h>
#endif

// C includes
#include <stdio.h>
#include <stdint.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// C++ includes
#include <exception>
#include <iostream>
#include <fstream>
#include <list>
//#include <thread>

// Application includes
#include "imageproccessing.h"
#include "initdepthsense.h"

/*
 * Set int kernel[9] to kernel defined by kern
 */
void pickKern(char* kern, int kernel[9]) 
{
    if (strncmp(kern, "edge", 4) == 0) 
        memcpy(kernel, edgeKern, 9*sizeof(int) );
    if (strncmp(kern, "shrp", 4) == 0) 
        memcpy(kernel, sharpKern, 9*sizeof(int) );
    if (strncmp(kern, "iden", 4) == 0) 
        memcpy(kernel, ogKern, 9*sizeof(int) );
    if (strncmp(kern, "blur", 4) == 0) 
        memcpy(kernel, blurKern, 9*sizeof(int) );
    if (strncmp(kern, "sobx", 4) == 0) 
        memcpy(kernel, sobelXKern, 9*sizeof(int) );
    if (strncmp(kern, "soby", 4) == 0) 
        memcpy(kernel, sobelYKern, 9*sizeof(int) );
    if (strncmp(kern, "scrx", 4) == 0) 
        memcpy(kernel, scharrXKern, 9*sizeof(int) );
    if (strncmp(kern, "scry", 4) == 0) 
        memcpy(kernel, scharrYKern, 9*sizeof(int) );
    if (strncmp(kern, "embs", 4) == 0) 
        memcpy(kernel, embossKern, 9*sizeof(int) );
    if (strncmp(kern, "edgh", 4) == 0) 
        memcpy(kernel, edgeHighKern, 9*sizeof(int) );
    if (strncmp(kern, "lapl", 4) == 0) 
        memcpy(kernel, lapKern, 9*sizeof(int) );
}

/*
 * Three convoloution implementations, must be way to factor this nicely 
 * (unique array types make it difficult) 
 */
int convolve(int i, int j, int kern[9], char *kernel, int W, int H, double bias, int jump) 
{
    int edge = 0; int w = 3;
    edge = edge + kern[1*w +1] * (int)diffMap[i*W*3 + j*3 + jump];
    // UP AND DOWN
    if (i - 1 > 0)
        edge = edge + kern[0*w + 1] * (int)diffMap[(i-1)*W*3 + j*3 + jump];
    else
        edge = edge + kern[0*w + 1] * (int)diffMap[(i-0)*W*3 + j*3 + jump]; // extend

    if (i + 1 < H)
        edge = edge + kern[2*w + 1] * (int)diffMap[(i+1)*W*3 + j*3 + jump];
    else
        edge = edge + kern[2*w + 1] * (int)diffMap[(i+0)*W*3 + j*3 + jump]; // extend

    // LEFT AND RIGHT
    if (j - 1 > 0)
        edge = edge + kern[1*w + 0] * (int)diffMap[i*W*3 + (j-1)*3 + jump]; 
    else                    
        edge = edge + kern[1*w + 0] * (int)diffMap[i*W*3 + (j-0)*3 + jump]; // extend

    if (j + 1 < W)         
        edge = edge + kern[1*w + 2] * (int)diffMap[i*W*3 + (j+1)*3 + jump]; 
    else                    
        edge = edge + kern[1*w + 2] * (int)diffMap[i*W*3 + (j+0)*3 + jump]; // extend
    
    // UP LEFT AND UP RIGHT
    if ((j - 1 > 0) && (i - 1) > 0)
        edge = edge + kern[0*w + 0] * (int)diffMap[(i-1)*W*3 + (j-1)*3 + jump]; 
    else                    
        edge = edge + kern[0*w + 0] * (int)diffMap[(i-0)*W*3 + (j-0)*3 + jump]; // extend

    if ((j + 1 < W) && (i - 1) > 0)
        edge = edge + kern[0*w + 2] * (int)diffMap[(i-1)*W*3 + (j+1)*3 + jump]; 
    else                     
        edge = edge + kern[0*w + 2] * (int)diffMap[(i-0)*W*3 + (j+0)*3 + jump]; // extend
    
    // DOWN LEFT AND DOWN RIGHT
    if ((j - 1 > 0) && (i + 1) < H)
        edge = edge + kern[2*w + 0] * (int)diffMap[(i+1)*W*3 + (j-1)*3 + jump]; 
    else                      
        edge = edge + kern[2*w + 0] * (int)diffMap[(i+0)*W*3 + (j-0)*3 + jump]; // extend

    if ((j + 1 < W) && (i + 1) < H)
        edge = edge + kern[2*w + 2] * (int)diffMap[(i+1)*W*3 + (j+1)*3 + jump]; 
    else                     
        edge = edge + kern[2*w + 2] * (int)diffMap[(i+0)*W*3 + (j+0)*3 + jump]; // extend
    
    edge = (edge * ((double)1 - bias)) + ((double)32000 * (bias));

    // clamp
    if (edge < 0)
        edge = 0;
    
    if (edge > 31999)
        edge = 31999;

    return edge;

}


int convolveDepth(int i, int j, int kern[9], char *kernel, int W, int H, double bias) 
{
    int edge = 0; int w = 3;
    edge = edge + kern[1*w +1] * (int)dConvolveMap[i*W + j];
    // UP AND DOWN
    if (i - 1 > 0)
        edge = edge + kern[0*w + 1] * (int)dConvolveMap[(i-1)*W + j];
    else
        edge = edge + kern[0*w + 1] * (int)dConvolveMap[(i-0)*W + j]; // extend

    if (i + 1 < H)
        edge = edge + kern[2*w + 1] * (int)dConvolveMap[(i+1)*W + j];
    else
        edge = edge + kern[2*w + 1] * (int)dConvolveMap[(i+0)*W + j]; // extend

    // LEFT AND RIGHT
    if (j - 1 > 0)
        edge = edge + kern[1*w + 0] * (int)dConvolveMap[i*W + (j-1)]; 
    else                    
        edge = edge + kern[1*w + 0] * (int)dConvolveMap[i*W + (j-0)]; // extend

    if (j + 1 < W)         
        edge = edge + kern[1*w + 2] * (int)dConvolveMap[i*W + (j+1)]; 
    else                    
        edge = edge + kern[1*w + 2] * (int)dConvolveMap[i*W + (j+0)]; // extend
    
    // UP LEFT AND UP RIGHT
    if ((j - 1 > 0) && (i - 1) > 0)
        edge = edge + kern[0*w + 0] * (int)dConvolveMap[(i-1)*W + (j-1)]; 
    else                    
        edge = edge + kern[0*w + 0] * (int)dConvolveMap[(i-0)*W + (j-0)]; // extend

    if ((j + 1 < W) && (i - 1) > 0)
        edge = edge + kern[0*w + 2] * (int)dConvolveMap[(i-1)*W + (j+1)]; 
    else                     
        edge = edge + kern[0*w + 2] * (int)dConvolveMap[(i-0)*W + (j+0)]; // extend
    
    // DOWN LEFT AND DOWN RIGHT
    if ((j - 1 > 0) && (i + 1) < H)
        edge = edge + kern[2*w + 0] * (int)dConvolveMap[(i+1)*W + (j-1)]; 
    else                      
        edge = edge + kern[2*w + 0] * (int)dConvolveMap[(i+0)*W + (j-0)]; // extend

    if ((j + 1 < W) && (i + 1) < H)
        edge = edge + kern[2*w + 2] * (int)dConvolveMap[(i+1)*W + (j+1)]; 
    else                     
        edge = edge + kern[2*w + 2] * (int)dConvolveMap[(i+0)*W + (j+0)]; // extend

    
    if (strncmp(kernel, "blur", 4) == 0) 
        edge = edge/(4+2+2+1+1+1+1);
    
    edge = (edge * ((double)1 - bias)) + ((double)32000 * (bias));

    // clamp
    if (edge < 0)
        edge = 0;
    
    if (edge > 31999)
        edge = 31999;

    return edge;

}

int convolveColour(int i, int j, int kern[9], char *kernel, int W, int H, double bias) 
{
    int edge = 0; int w = 3;
    edge = edge + kern[1*w +1] * (int)cConvolveMap[i*W + j];
    // UP AND DOWN
    if (i - 1 > 0)
        edge = edge + kern[0*w + 1] * (int)cConvolveMap[(i-1)*W + j];
    else
        edge = edge + kern[0*w + 1] * (int)cConvolveMap[(i-0)*W + j]; // extend

    if (i + 1 < H)
        edge = edge + kern[2*w + 1] * (int)cConvolveMap[(i+1)*W + j];
    else
        edge = edge + kern[2*w + 1] * (int)cConvolveMap[(i+0)*W + j]; // extend

    // LEFT AND RIGHT
    if (j - 1 > 0)
        edge = edge + kern[1*w + 0] * (int)cConvolveMap[i*W + (j-1)]; 
    else                    
        edge = edge + kern[1*w + 0] * (int)cConvolveMap[i*W + (j-0)]; // extend

    if (j + 1 < W)         
        edge = edge + kern[1*w + 2] * (int)cConvolveMap[i*W + (j+1)]; 
    else                    
        edge = edge + kern[1*w + 2] * (int)cConvolveMap[i*W + (j+0)]; // extend
    
    // UP LEFT AND UP RIGHT
    if ((j - 1 > 0) && (i - 1) > 0)
        edge = edge + kern[0*w + 0] * (int)cConvolveMap[(i-1)*W + (j-1)]; 
    else                    
        edge = edge + kern[0*w + 0] * (int)cConvolveMap[(i-0)*W + (j-0)]; // extend

    if ((j + 1 < W) && (i - 1) > 0)
        edge = edge + kern[0*w + 2] * (int)cConvolveMap[(i-1)*W + (j+1)]; 
    else                     
        edge = edge + kern[0*w + 2] * (int)cConvolveMap[(i-0)*W + (j+0)]; // extend
    
    // DOWN LEFT AND DOWN RIGHT
    if ((j - 1 > 0) && (i + 1) < H)
        edge = edge + kern[2*w + 0] * (int)cConvolveMap[(i+1)*W + (j-1)]; 
    else                      
        edge = edge + kern[2*w + 0] * (int)cConvolveMap[(i+0)*W + (j-0)]; // extend

    if ((j + 1 < W) && (i + 1) < H)
        edge = edge + kern[2*w + 2] * (int)cConvolveMap[(i+1)*W + (j+1)]; 
    else                     
        edge = edge + kern[2*w + 2] * (int)cConvolveMap[(i+0)*W + (j+0)]; // extend
    

    if (strncmp(kernel, "blur", 4) == 0) 
        edge = edge/(4+2+2+1+1+1+1);

    edge = (edge * ((double)1 - bias)) + ((double)32000 * (bias));

    // clamp
    if (edge < 0)
        edge = 0;
    
    if (edge > 31999)
        edge = 31999;

    return edge;

}

/*
 * Loop and convolve certain maps
 */
void applyKernelDepth(char *kern, int W, int H, double bias) 
{
    int kernel[9]; pickKern(kern, kernel);
    memset(dConvolveResult, 32002, sizeof(dConvolveResult));
    for(int i=0; i < H; i++) {
        for(int j=0; j < W; j++) {
            dConvolveResult[i*W + j] = convolveDepth(i,j, kernel, kern, W, H, bias);
        }
    }

}

void applyKernelColour(char *kern, int W, int H, double bias) 
{
    int kernel[9]; pickKern(kern, kernel);
    memset(cConvolveResult, 0, sizeof(cConvolveResult));

    // saftey
    if (bias > 1)
        bias = 0;

    for(int i=0; i < H; i++) {
        for(int j=0; j < W; j++) {
            cConvolveResult[i*W + j] = convolveColour(i,j, kernel, kern, W, H, bias);
        }
    }

}

void applyKernel3D(char *kern, int W, int H, double bias) 
{
    int kernel[9]; pickKern(kern, kernel);
    memset(diffResult, 32002, dshmsz);
    
    for(int i=0; i < H; i++) {
        for(int j=0; j < W; j++) {
            diffResult[i*W*3 + j*3 + 0] = convolve(i,j, kernel, kern, W, H, bias, 0);
            diffResult[i*W*3 + j*3 + 1] = convolve(i,j, kernel, kern, W, H, bias, 1);
            diffResult[i*W*3 + j*3 + 2] = convolve(i,j, kernel, kern, W, H, bias, 2);
        }
    }

}

/* 
 * Compute gradient of the vertex map
 */
void computeDifferential(char *kern, double bias) 
{
    memset(dxMap, 0, dshmsz);
    memset(dyMap, 0, dshmsz);
    memcpy(diffMap, normalMap, dshmsz*3);

    (void) kern;

    // compute dxMap
    applyKernel3D((char*)"sobx", dW, dH, bias);
    memcpy(dxMap, diffResult, dshmsz*3);

    // compute dyMap
    applyKernel3D((char*)"soby", dW, dH, bias);
    memcpy(dyMap, diffResult, dshmsz*3);

}

/*
 * Compute the normal Map
 */
void crossMaps() 
{
    int16_t dxx; int16_t dxy; int16_t dxz; 
    int16_t dyx; int16_t dyy; int16_t dyz; 

    double nx; double ny; double nz; 
    double length;
    for(int i=0; i < dH; i++) {
        for(int j=0; j < dW; j++) {
            dxx = dxMap[i*dW*3 + j*3 + 0];
            dxy = dxMap[i*dW*3 + j*3 + 1];
            dxz = dxMap[i*dW*3 + j*3 + 2];

            dyx = dyMap[i*dW*3 + j*3 + 0];
            dyy = dyMap[i*dW*3 + j*3 + 1];
            dyz = dyMap[i*dW*3 + j*3 + 2];
           
            nx = dxy*dyz - dxz*dyy; 
            ny = dxz*dyx - dxx*dyz;
            nz = dxx*dyy - dxy*dyx;

            length = sqrt(nx*nx + ny*ny + nz*nz);

            normalResult[i*dW*3 + j*3 + 0] = nx/length * 65535;
            normalResult[i*dW*3 + j*3 + 1] = ny/length * 65535;
            normalResult[i*dW*3 + j*3 + 2] = nz/length * 65535;
        }
    }
}

/*
 * Convert colour image to greyscale
 */
void toGreyScale(double rweight, double gweight, double bweight) 
{
    uint8_t red; uint8_t green; uint8_t blue;
    for(int i=0; i < cH; i++) {
        for(int j=0; j < cW; j++) {
            red =   greyColourMap[i*cW*3 + j*3 + 0];
            green = greyColourMap[i*cW*3 + j*3 + 1];
            blue =  greyColourMap[i*cW*3 + j*3 + 2];
            greyResult[i*cW + j] = (uint8_t)(red*rweight + green*gweight + blue*bweight);
        }
    }
}


