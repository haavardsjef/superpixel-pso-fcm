package no.haavardsjef.classification;

import libsvm.*;
import lombok.extern.log4j.Log4j2;

@Log4j2
public class SVMParameterSearch {

	private static final double[] GAMMA_RANGE = {0.1, 0.5, 1, 2, 5};
	private static final double[] C_RANGE = {0.1, 1, 10, 100, 1000};
	private static final int K_FOLDS = 5;

	public static svm_parameter findBestParameters(svm_problem problem) {
		log.info("Performing grid search with {} samples and {} folds", problem.l, K_FOLDS);
		double bestAccuracy = 0;
		double bestGamma = 0;
		double bestC = 0;

		for (double gamma : GAMMA_RANGE) {
			for (double C : C_RANGE) {
				svm_parameter param = new svm_parameter();
				param.svm_type = svm_parameter.C_SVC;
				param.kernel_type = svm_parameter.RBF;
				param.gamma = gamma;
				param.C = C;
				param.eps = 0.001;
				param.cache_size = 100;

				double accuracy = performCrossValidation(problem, param, K_FOLDS);
				if (accuracy > bestAccuracy) {
					bestAccuracy = accuracy;
					bestGamma = gamma;
					bestC = C;
				}
			}
		}

		svm_parameter bestParam = new svm_parameter();
		bestParam.svm_type = svm_parameter.C_SVC;
		bestParam.kernel_type = svm_parameter.RBF;
		bestParam.gamma = bestGamma;
		bestParam.C = bestC;
		bestParam.eps = 0.001;
		bestParam.cache_size = 100;

		return bestParam;
	}

	private static double performCrossValidation(svm_problem problem, svm_parameter param, int kFolds) {
		svm.svm_set_print_string_function(new svm_print_interface() {
			@Override
			public void print(String s) {
				// Do nothing, effectively muting the libsvm output
			}
		});

		double[] target = new double[problem.l];
		svm.svm_cross_validation(problem, param, kFolds, target);

		int correct = 0;
		for (int i = 0; i < problem.l; i++) {
			if (target[i] == problem.y[i]) {
				correct++;
			}
		}

		return (double) correct / problem.l;
	}


}
