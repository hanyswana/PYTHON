// NormManhLib.h

#ifndef NormManhLib_h
#define NormManhLib_h

#include <Arduino.h>

class NormManhLib {
public:
    static void normalize_with_manhattan(float* arr, int length);
private:
    static float manhattan_norm(float* arr, int length);
};

#endif