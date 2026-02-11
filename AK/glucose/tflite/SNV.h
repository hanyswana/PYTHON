#ifndef SNV_H
#define SNV_H

#include <Arduino.h>

class SNV {
public:
    void preprocess(float* arr, int length);

private:
    float calculateMean(float* arr, int length);
    float calculateStdDev(float* arr, int length, float mean);
};

#endif
