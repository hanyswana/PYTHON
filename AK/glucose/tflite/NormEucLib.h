// NormEucLib.h

#ifndef NormEucLib_h
#define NormEucLib_h

#include <Arduino.h>

class NormEucLib {
public:
    static void normalize_with_euclidean(float* arr, int length);
private:
    static float euclidean_norm(float* arr, int length);
};

#endif