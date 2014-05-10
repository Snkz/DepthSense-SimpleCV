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

// C++ includes
#include <exception>
#include <iostream>
#include <fstream>
#include <list>
//#include <thread>

// Application includes
#include "imageproccessing.h"
#include "imagebuffers.h"

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
int convolve(int i, int j, int kern[9], char *kernel, int W, int H, double bias) 
{
    int edge = 0; int w = 3;
    edge = edge + kern[1*w +1] * (int)diffMap[i*W + j];
    // UP AND DOWN
    if (i - 1 > 0)
        edge = edge + kern[0*w + 1] * (int)diffMap[(i-1)*W + j];
    else
        edge = edge + kern[0*w + 1] * (int)diffMap[(i-0)*W + j]; // extend

    if (i + 1 < H)
        edge = edge + kern[2*w + 1] * (int)diffMap[(i+1)*W + j];
    else
        edge = edge + kern[2*w + 1] * (int)diffMap[(i+0)*W + j]; // extend

    // LEFT AND RIGHT
    if (j - 1 > 0)
        edge = edge + kern[1*w + 0] * (int)diffMap[i*W + (j-1)]; 
    else                    
        edge = edge + kern[1*w + 0] * (int)diffMap[i*W + (j-0)]; // extend

    if (j + 1 < W)         
        edge = edge + kern[1*w + 2] * (int)diffMap[i*W + (j+1)]; 
    else                    
        edge = edge + kern[1*w + 2] * (int)diffMap[i*W + (j+0)]; // extend
    
    // UP LEFT AND UP RIGHT
    if ((j - 1 > 0) && (i - 1) > 0)
        edge = edge + kern[0*w + 0] * (int)diffMap[(i-1)*W + (j-1)]; 
    else                    
        edge = edge + kern[0*w + 0] * (int)diffMap[(i-0)*W + (j-0)]; // extend

    if ((j + 1 < W) && (i - 1) > 0)
        edge = edge + kern[0*w + 2] * (int)diffMap[(i-1)*W + (j+1)]; 
    else                     
        edge = edge + kern[0*w + 2] * (int)diffMap[(i-0)*W + (j+0)]; // extend
    
    // DOWN LEFT AND DOWN RIGHT
    if ((j - 1 > 0) && (i + 1) < H)
        edge = edge + kern[2*w + 0] * (int)diffMap[(i+1)*W + (j-1)]; 
    else                      
        edge = edge + kern[2*w + 0] * (int)diffMap[(i+0)*W + (j-0)]; // extend

    if ((j + 1 < W) && (i + 1) < H)
        edge = edge + kern[2*w + 2] * (int)diffMap[(i+1)*W + (j+1)]; 
    else                     
        edge = edge + kern[2*w + 2] * (int)diffMap[(i+0)*W + (j+0)]; // extend
    
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
            diffResult[i*W + j] = convolve(i,j, kernel, kern, W, H, bias);
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
    memset(dzMap, 0, dshmsz);

    (void) kern;

    int16_t x; int16_t y; int16_t z;
    for(int i=0; i < dH; i++) {
        for(int j=0; j < dW; j++) {
            x = normalMap[i*dW*3 + j*3 + 0];
            y = normalMap[i*dW*3 + j*3 + 1];
            z = normalMap[i*dW*3 + j*3 + 2];
            if (z != 32001) {
                dxMap[i*dW + j] = x;
                dyMap[i*dW + j] = y;
                dzMap[i*dW + j] = z;
            }
        }
    }
 
    // compute dxMap
    memcpy(diffMap, dzMap, dshmsz);
    applyKernel3D((char*)"sobx", dW, dH, bias);
    memcpy(dxMap, diffResult, dshmsz);

    // compute dyMap
    memcpy(diffMap, dzMap, dshmsz);
    applyKernel3D((char*)"soby", dW, dH, bias);
    memcpy(dyMap, diffResult, dshmsz);

    // compute dzMap
    //memcpy(diffMap, dzMap, dshmsz);
    //applyKernel3D(kern, dW, dH, bias); 
    //memcpy(dzMap, diffResult, dshmsz);
}

/*
 * Compute the normal Map
 */
void crossMaps() 
{
    int16_t x; int16_t y; int16_t z;
    for(int i=0; i < dH; i++) {
        for(int j=0; j < dW; j++) {
            x = dxMap[i*dW + j];
            y = dyMap[i*dW + j];
            z = dzMap[i*dW + j];
            
            normalResult[i*dW*3 + j*3 + 0] = (x);
            normalResult[i*dW*3 + j*3 + 1] = (y);
            normalResult[i*dW*3 + j*3 + 2] = (255);
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

/*
 * Build a colour image with points defined in the depthmap
 */
void buildSyncMap()
{
    int ci, cj;
    uint8_t colx;
    uint8_t coly;
    uint8_t colz;
    float uvx;
    float uvy;

    for(int i=0; i < dH; i++) {
        for(int j=0; j < dW; j++) {
            uvx = uvMapClone[i*dW*2 + j*2 + 0];    
            uvy = uvMapClone[i*dW*2 + j*2 + 1];    
            colx = 0;
            coly = 0;
            colz = 0;
            
            if((uvx > 0 && uvx < 1 && uvy > 0 && uvy < 1) && 
                (depthMapClone[i*dW + j] < 32000)){
                ci = (int) (uvy * ((float) cH));
                cj = (int) (uvx * ((float) cW));
                colx = colourMapClone[ci*cW*3 + cj*3 + 0];
                coly = colourMapClone[ci*cW*3 + cj*3 + 1];
                colz = colourMapClone[ci*cW*3 + cj*3 + 2];
            }
          
            syncMapClone[i*dW*3 + j*3 + 0] = colx;
            syncMapClone[i*dW*3 + j*3 + 1] = coly;
            syncMapClone[i*dW*3 + j*3 + 2] = colz;

        }
    }
}




