package no.haavardsjef.classification;

import libsvm.*;
import lombok.extern.log4j.Log4j2;
import no.haavardsjef.Dataset;
import no.haavardsjef.utility.DataLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

@Log4j2
public class SVMClassifier implements IClassifier {

	Dataset dataset;

	public SVMClassifier(Dataset dataset) {
		this.dataset = dataset;
	}


	public void evaluate(List<Integer> selectedBands) {
		log.info("Evaluating SVM classifier with selected bands: " + selectedBands);
		// TODO: Consider normalizing each pixel so that all bands add to 1


		int[] groundTruth = dataset.getGroundTruthFlattenedAsArray();
		double[][] pixelValuesForSelectedBands = dataset.getBandsFlattened(selectedBands).transpose().toDoubleMatrix();


		if (pixelValuesForSelectedBands.length != groundTruth.length) {
			throw new RuntimeException("The number of ground truths does not match the number of pixels");
		}

		Sample[] samples = new Sample[groundTruth.length];

		for (int i = 0; i < pixelValuesForSelectedBands.length; i++) {
			samples[i] = new Sample(i, groundTruth[i], pixelValuesForSelectedBands[i]);
		}

		// Split 10% of the data for testing
		int testSize = (int) (samples.length * 0.1);
		Sample[] testSamples = new Sample[testSize];
		Sample[] trainingSamples = new Sample[samples.length - testSize];

		System.arraycopy(samples, 0, trainingSamples, 0, trainingSamples.length);
		System.arraycopy(samples, trainingSamples.length, testSamples, 0, testSamples.length);

		svm_model model = train(trainingSamples);
		test(model, testSamples);


	}

	private svm_model train(Sample[] data) {
		System.out.println("Starting training");
		svm_problem trainingProblem = createProblem(data);

		svm_parameter params = new svm_parameter();
		params.svm_type = svm_parameter.C_SVC;
		params.kernel_type = svm_parameter.RBF;
		params.degree = 3;
		params.coef0 = 0;
		params.nu = 0.5;
		params.cache_size = 100;
		params.eps = 1e-3;
		params.p = 0.1;
		params.shrinking = 1;
		params.probability = 0;
		params.nr_weight = 0;
		params.weight_label = new int[0];
		params.weight = new double[0];

		params.gamma = 241.63182404651315;
		params.C = 2.0;

		svm_model model = svm.svm_train(trainingProblem, params);

		return model;
	}

	private void test(svm_model model, Sample[] testSamples) {
		System.out.println("Starting testing");

		svm_problem problem = createProblem(testSamples);

		int correct = 0;

		for (int i = 0; i < problem.l; i++) {
			double predictedLabel = svm.svm_predict(model, problem.x[i]);
			double actualLabel = problem.y[i];

			System.out.println("Predicted: " + predictedLabel + " Actual: " + actualLabel);

			if (predictedLabel == actualLabel) {
				correct++;
			}

		}

		System.out.println("Correct: " + correct + " Total: " + problem.l);
		System.out.println("Accuracy: " + (double) correct / problem.l);
	}

	private svm_problem createProblem(Sample[] data) {
		svm_problem prob = new svm_problem();

		int numFeatures = data[0].features().length;

		prob.l = data.length;
		prob.x = new svm_node[prob.l][];
		prob.y = new double[prob.l];

		for (int i = 0; i < prob.l; i++) {
			prob.y[i] = data[i].label();

			prob.x[i] = new svm_node[numFeatures];

			for (int j = 0; j < numFeatures; j++) {
				svm_node node = new svm_node();
				node.index = j + 1;
				node.value = data[i].features()[j];

				prob.x[i][j] = node;
			}
		}

		return prob;
	}

}
