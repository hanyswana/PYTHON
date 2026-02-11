# import joblib
#
# # Load the model
# model_path = 'C:/Users/hanys.harun/PycharmProjects/pythonprojectTF/SBH-PLSR/model/Best_PLSR_model_sbh-1_HW_br.joblib'
# model = joblib.load(model_path)
#
# # Assuming the model is directly a PLSR model or the last step of a pipeline
# if hasattr(model, 'coef_'):
#     coefficients = model.coef_
#     intercept = model.intercept_
# else:  # If the model is part of a pipeline
#     plsr_model = model.named_steps['plsr']  # Adjust 'plsr' based on actual step name
#     coefficients = plsr_model.coef_
#     intercept = plsr_model.intercept_
#
# # Print coefficients and intercept
# print("Coefficients:", coefficients)
# print("Intercept:", intercept)


# Coefficients and intercept from your model
coefficients = [-0.526570423, -1.91212719, -7.33547422, -2.99806115,
                -0.536203934, 15.3347324, 1.14988866, -1.78547162,
                1.06499975, 0.000166103678, -5.89210764, -0.817553381,
                -1.19542551, -10.2002520, 13.9880525, -27.5304279,
                12.1718842, -12.6008738, 12.7738076]
intercept = 32.14666667

# Scale means and standard deviations
scale_means = [-0.0386063, 0.53431707, 0.26560835, 0.24589176, -0.04954516, 0.35985291,
               0.20759762, -0.155009, -0.01100725, -0.06199716, -0.03001665, -0.1209569,
               -0.04268552, -0.10975196, -0.17307526, -0.25537868, -0.29752406, -0.13602773,
               -0.1316861]
scale_stds = [0.02732845, 0.17164621, 0.09682912, 0.13354172, 0.0337835, 0.15588749,
              0.08766698, 0.03725253, 0.05592834, 0.01389148, 0.02935259, 0.03716613,
              0.06520373, 0.0557053, 0.06903064, 0.07639655, 0.08253501, 0.12597811,
              0.11961613]

# Scaled input data calculation
input_data = [-0.038097385, 0.674178086, 0.363047764, 0.395696685, 0.008994345,
              0.55853516, 0.325761445, -0.148518105, -0.044105426, -0.063470254,
              0.001362715, -0.172086639, -0.05487497, -0.166334284, -0.25773787,
              -0.406314442, -0.456188296, -0.272299885, -0.247548644]

scaled_data = [(input_data[i] - scale_means[i]) / scale_stds[i] for i in range(19)]

# Prediction calculation
prediction = sum(c * x for c, x in zip(coefficients, scaled_data)) + intercept

# Output the prediction
print(f"The manually calculated prediction is: {prediction}")

