#include "SNV.h"
#include <Arduino.h>
#include <math.h>

void SNV::preprocess(float* arr, int length) {
    float mean = calculateMean(arr, length);
    float stdDev = calculateStdDev(arr, length, mean);

    for (int i = 0; i < length; i++) {
        arr[i] = (arr[i] - mean) / stdDev;
    }
}

float SNV::calculateMean(float* arr, int length) {
    float sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += arr[i];
    }
    return sum / length;
}

float SNV::calculateStdDev(float* arr, int length, float mean) {
    float sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += pow(arr[i] - mean, 2);
    }
    return sqrt(sum / length);
}
