#include "savgol.h"
#include <vector>
#include <stdexcept>

SavGol::SavGol(int window_size, int poly_order, int deriv_order)
: window_size(window_size), poly_order(poly_order), deriv_order(deriv_order) {
    calculateCoefficients();
}

std::vector<double> SavGol::mirrorPad(const std::vector<double>& y) {
    int half_window = (window_size - 1) / 2;
    std::vector<double> padded(y.size() + 2 * half_window);

    for (int i = 0; i < half_window; ++i) {
        padded[half_window - i - 1] = y[i + 1];
        padded[padded.size() - half_window + i] = y[y.size() - 2 - i];
    }

    for (size_t i = 0; i < y.size(); ++i) {
        padded[half_window + i] = y[i];
    }

    return padded;
}

void SavGol::calculateCoefficients() {
    // Using the provided coefficients for each derivative order
    if (deriv_order == 0) {
        coefficients = {-0.08571429, 0.34285714, 0.48571429, 0.34285714, -0.08571429};
    } else if (deriv_order == 1) {
        coefficients = {-0.2, -0.1, -1.11022302e-16, 0.1, 0.2};
    } else if (deriv_order == 2) {
        coefficients = {0.28571429, -0.14285714, -0.28571429, -0.14285714, 0.28571429};
    }
}

std::vector<double> SavGol::applyFilter(const std::vector<double>& y) {
    std::vector<double> padded = mirrorPad(y);
    std::vector<double> result(y.size(), 0.0);

    for (size_t i = 0; i < y.size(); ++i) {
        for (int j = 0; j < window_size; ++j) {
            result[i] += coefficients[j] * padded[i + j];
        }
    }

    return result;
}
