/*
 * DepthSense SDK for Python and SimpleCV
 * -----------------------------------------------------------------------------
 * file:            ginac.cxx
 * author:          Abdi Dahir
 * modified:        May 9 2014
 * vim:             set fenc=utf-8:ts=4:sw=4:expandtab:
 * 
 * Linear algebra ops for image proccessing is done here
 * -----------------------------------------------------------------------------
 */

// MS completly untested
#ifdef _MSC_VER
#include <windows.h>
#endif

// C++ includes
#include <iostream>
#include <ginac/ginac.h>

// Application includes
#include "initdepthsense.h"
using namespace std;
using namespace GiNaC;

// assume the index provided is the start of a valid 3x3 map
// compute the normal according to that stackoverflow post
// ginac error handling is strange and the return type is numeric for its meths
// NOT int/double/float etc
// NOTE: ginac imports its own math lib (CNL or something) could cause issues
int compute_normal(int I, int J, int coef[]) {
    symbol x2("x1"), x1("x2"), x3("x3");
    
    matrix A(3,3);
    matrix B(3,1);
    matrix X(3,1);
    
    int x_sqr = 0; int y_sqr = 0; int xy = 0; int x = 0; int y = 0;
    int xz = 0; int yz = 0; int z = 0;
    int t_x; int t_y; int t_z;

    for(int i=I; i < I + 3; i++) {
        for(int j=J; j < J + 3; j++) {
            
            t_x = diffMap[i*dW*3 + j*3 + 0];
            t_y = diffMap[i*dW*3 + j*3 + 1];
            t_z = diffMap[i*dW*3 + j*3 + 2];

            x_sqr += t_x * t_x;
            y_sqr += t_y * t_y;

            x += t_x;
            y += t_y;
            z += t_z;

            xy += t_x * t_y;
            xz += t_x * t_z;
            yz += t_y * t_z;
        }
    }
    
    A = x_sqr, xy, x,
        xy, y_sqr, y,
        x, y, 9;

    B = xz, yz, z;

    X = x1, x2, x3;

    try {
        matrix result = A.solve(X,B);
        cout << result << endl;
        // figure out how to read value from matrix
        coef[0] = result(0,0).to_int(); coef[1] = result(1,0).to_int(); coef[2] = result(2,0).to_int(); 
    } catch (exception &p) {
        cerr << p.what() << endl;
        coef[0] = 0; coef[1] = 0; coef[2] = 0; 
        return 1;
    } 

    return 0;
}
