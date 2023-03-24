package no.haavardsjef.classification;

import libsvm.*;
import lombok.extern.log4j.Log4j2;
import no.haavardsjef.Dataset;
import no.haavardsjef.utility.DataLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

@Log4j2
public class SVMClassifier implements IClassifier {

	Dataset dataset;

	public SVMClassifier(Dataset dataset) {
		this.dataset = dataset;

		svm.svm_set_print_string_function(new libsvm.svm_print_interface() {
			@Override
			public void print(String s) {
			} // Disables svm output
		});
	}


	public void evaluate(List<Integer> selectedBands) {
		log.info("Evaluating SVM classifier with selected bands: " + selectedBands);
		// TODO: Consider normalizing each pixel so that all bands add to 1


		// Load features and ground truth
		int[] groundTruth = dataset.getGroundTruthFlattenedAsArray();
		double[][] pixelValuesForSelectedBands = dataset.getBandsFlattened(selectedBands).transpose().toDoubleMatrix();

		// Verify that the number of ground truths matches the number of pixels
		if (pixelValuesForSelectedBands.length != groundTruth.length) {
			throw new RuntimeException("The number of ground truths does not match the number of pixels");
		}

		// Create samples
		Sample[] samples = new Sample[groundTruth.length];
		for (int i = 0; i < pixelValuesForSelectedBands.length; i++) {
			samples[i] = new Sample(i, groundTruth[i], pixelValuesForSelectedBands[i]);
		}

		// Shuffle and split into training and test set
		double trainingRatio = 0.8; // 80% for training, 20% for testing
		Sample[][] split = splitSamples(samples, trainingRatio);
		Sample[] trainingSamples = split[0];
		Sample[] testSamples = split[1];

		svm_model model = train(trainingSamples);

		double accuracy = evaluateAccuracy(model, testSamples);


	}

	public static void saveConfusionMatrixToCSV(int[][] confusionMatrix, String filePath) {
		try (PrintWriter writer = new PrintWriter(new File(filePath))) {
			for (int i = 0; i < confusionMatrix.length; i++) {
				StringBuilder row = new StringBuilder();
				for (int j = 0; j < confusionMatrix[i].length; j++) {
					row.append(confusionMatrix[i][j]);
					if (j < confusionMatrix[i].length - 1) {
						row.append(",");
					}
				}
				writer.println(row.toString());
			}
		} catch (FileNotFoundException e) {
			System.err.println("Error saving confusion matrix to CSV file: " + e.getMessage());
		}
	}

	private svm_model train(Sample[] data) {
		log.info("Training SVM classifier with " + data.length + " samples");
		long startTime = System.currentTimeMillis();
		svm_problem trainingProblem = createProblem(data);

		svm_parameter param = new svm_parameter();
		param.svm_type = svm_parameter.C_SVC;
		param.kernel_type = svm_parameter.RBF;
		param.gamma = 0.5;
		param.C = 1;
		param.eps = 0.001;
		param.cache_size = 100;

		svm_model model = svm.svm_train(trainingProblem, param);
		long endTime = System.currentTimeMillis();
		log.info("Training took " + (endTime - startTime) / 1000 + " seconds");

		return model;
	}


	private static int predict(svm_model model, double[] features) {
		svm_node[] nodes = new svm_node[features.length];
		for (int i = 0; i < features.length; i++) {
			nodes[i] = new svm_node();
			nodes[i].index = i + 1;
			nodes[i].value = features[i];
		}

		double prediction = svm.svm_predict(model, nodes);
		return (int) prediction;
	}

	private static double evaluateAccuracy(svm_model model, Sample[] testSamples) {
		log.info("Evaluating accuracy of SVM classifier with " + testSamples.length + " samples");
		long startTime = System.currentTimeMillis();
		int numCorrectPredictions = 0;
		int numClasses = Arrays.stream(testSamples).mapToInt(Sample::label).max().getAsInt() + 1;
		int[][] confusionMatrix = new int[numClasses][numClasses];

		for (Sample sample : testSamples) {
			int trueLabel = sample.label();
			double[] features = sample.features();
			int predictedLabel = predict(model, features);

			if (predictedLabel == trueLabel) {
				numCorrectPredictions++;
			}
			confusionMatrix[trueLabel][predictedLabel]++;
		}

		String filePath = "confusion_matrix.csv";
		saveConfusionMatrixToCSV(confusionMatrix, filePath);
		log.info("Confusion matrix saved to " + filePath);

		double accuracy = (double) numCorrectPredictions / testSamples.length;
		long endTime = System.currentTimeMillis();
		log.info("Evaluation took " + (endTime - startTime) + " ms");
		log.info("Accuracy: " + accuracy * 100 + "%");
		log.info("Number of correct predictions: " + numCorrectPredictions, " out of " + testSamples.length);
		return accuracy;
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

	private static Sample[][] splitSamples(Sample[] samples, double trainingRatio) {
		// Shuffle the samples array
		Random random = new Random();
		for (int i = samples.length - 1; i > 0; i--) {
			int index = random.nextInt(i + 1);
			Sample temp = samples[index];
			samples[index] = samples[i];
			samples[i] = temp;
		}

		// Split the samples array into training and test sets
		int trainSize = (int) (samples.length * trainingRatio);
		int testSize = samples.length - trainSize;
		Sample[] trainingSamples = new Sample[trainSize];
		Sample[] testSamples = new Sample[testSize];
		System.arraycopy(samples, 0, trainingSamples, 0, trainSize);
		System.arraycopy(samples, trainSize, testSamples, 0, testSize);

		return new Sample[][]{trainingSamples, testSamples};
	}


}
