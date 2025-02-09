

First, in order to run generate data necessary for the experiments, run the generate_data file.

In order to run the Gamma Poisson experiments, run the gamma_poisson_improved file.

In order to run the Weibull Inverse Gamma experiments, run the weibull_inverse_gamma_improved_improved file.

In order to produce figure 3 (the first logistic regression with t prior experiment), run the t_logistic_sample_size_check_large file, then the t_logistic_bounds_into_file_large file. Then run stan_experiments and then, finally t_logistic_plots1.

In order to produce figure 4 (the second logistic regression with t prior experiment), run t_logistic_sample_size_check_zero and then t_logistic_bounds_into_file_zero, setting the value of d to 1, 3, 5, 7, 9. Then, run t_logistic_plots2.

In order to produce figure 5 (the third logistic regression with t prior experiment), run t_logistic_sample_size_check_zero and then t_logistic_plots3.

In order to produce figure 6 (the fourth logistic regression with t prior experiment), run t_logistic_sample_size_check_large, t_logistic_sample_size_check_medium and t_logistic_sample_size_check_zero. Then run t_logistic_bounds_into_file_large, t_logistic_bounds_into_file_medium, t_logistic_bounds_into_file_zero, setting d=5. Then, run t_logistic_plots4. Finally, run t_logistic_condition_numbers.
