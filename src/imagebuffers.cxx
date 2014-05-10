/*
 * DepthSense SDK for Python and SimpleCV
 * -----------------------------------------------------------------------------
 * file:            imagebuffers.h                                              
 * author:          Abdi Dahir                           
 * modified:        May 9 2014                                               
 * vim:             set fenc=utf-8:ts=4:sw=4:expandtab:                      
 *                                                                             
 * Imagebuffers defined here for various image proccessing operations
 * -----------------------------------------------------------------------------
 */

#include <stdint.h>

// internal map copies
uint8_t colourMapClone[640*480*3];
int16_t depthMapClone[320*240];
int16_t vertexMapClone[320*240*3];
float accelMapClone[3];
float uvMapClone[320*240*2];
float vertexFMapClone[320*240*3];
uint8_t syncMapClone[320*240*3];
int16_t printMap[320*240*3];

// colouring depth map
int16_t depthCMap[320*240];
uint8_t depthColouredMap[320*240*3];
uint8_t depthColouredMapClone[320*240*3];

// internal maps for edge finding
int16_t dConvolveMap[320*240];
int16_t dConvolveResult[320*240];
int16_t dConvolveResultClone[320*240];

// internal maps for edge finding in colour
uint8_t cConvolveMap[640*480];
uint8_t cConvolveResult[640*480];
uint8_t cConvolveResultClone[640*480];

// internal maps for edge finding
uint8_t greyColourMap[640*480*3];
uint8_t greyResult[640*480];
uint8_t greyResultClone[640*480];

// internal maps for normal computation
int16_t normalMap[320*240*3];
int16_t dxMap[320*240];
int16_t dyMap[320*240];
int16_t dzMap[320*240];
int16_t diffMap[320*240];
int16_t diffResult[320*240];
int16_t normalResult[320*240*3];
int16_t normalResultClone[320*240*3];
