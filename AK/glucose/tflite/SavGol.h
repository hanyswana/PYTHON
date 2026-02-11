#ifndef SAVGOL_H
#define SAVGOL_H

#include <vector>

class SavGol {
public:
    SavGol(int window_size, int poly_order, int deriv_order);
    std::vector<double> applyFilter(const std::vector<double>& y);

private:
    int window_size;
    int poly_order;
    int deriv_order;
    std::vector<double> coefficients;

    void calculateCoefficients();
    std::vector<double> mirrorPad(const std::vector<double>& y);
};

#endif // SAVGOL_H
