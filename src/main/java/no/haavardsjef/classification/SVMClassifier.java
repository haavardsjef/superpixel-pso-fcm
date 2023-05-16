package no.haavardsjef.classification;

import libsvm.*;
import lombok.extern.log4j.Log4j2;
import no.haavardsjef.dataset.Dataset;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static no.haavardsjef.classification.ClassificationUtilities.*;

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

	public ClassificationResult evaluate(List<Integer> selectedBands, int numClassificationRuns) {
		return evaluate(selectedBands, numClassificationRuns, 0.1);
	}

	public void justTrain(List<Integer> selectedBands, int numTrainingRuns, double trainingRatio) {
		// Load features and ground truth
		int[] groundTruth = dataset.getGroundTruthFlattenedAsArray();
		double[][] pixelValuesForSelectedBands = dataset.getBandsFlattened(selectedBands).transpose().toDoubleMatrix();

		// Count number of classes
		int numClasses = Arrays.stream(groundTruth).max().getAsInt() + 1;


		// Verify that the number of ground truths matches the number of pixels
		if (pixelValuesForSelectedBands.length != groundTruth.length) {
			throw new RuntimeException("The number of ground truths does not match the number of pixels");
		}

		// Create samples
		Sample[] samples = new Sample[groundTruth.length];
		for (int i = 0; i < pixelValuesForSelectedBands.length; i++) {
			samples[i] = new Sample(i, groundTruth[i], pixelValuesForSelectedBands[i]);
		}

		// Run training runs
		for (int i = 0; i < numTrainingRuns; i++) {


			// Shuffle and split into training and test set
			Sample[][] split = splitSamples(samples, trainingRatio);
			Sample[] trainingSamples = split[0];
			Sample[] testSamples = split[1];


			svm_model model = trainWithoutGridSearch(trainingSamples);
		}

	}

	public ClassificationResult evaluate(List<Integer> selectedBands, int numClassificationRuns, double trainingRatio) {
		log.info("Evaluating SVM classifier with selected bands: " + selectedBands);


		// Load features and ground truth
		int[] groundTruth = dataset.getGroundTruthFlattenedAsArray();
		double[][] pixelValuesForSelectedBands = dataset.getBandsFlattened(selectedBands).transpose().toDoubleMatrix();

		// Count number of classes
		int numClasses = Arrays.stream(groundTruth).max().getAsInt() + 1;


		// Verify that the number of ground truths matches the number of pixels
		if (pixelValuesForSelectedBands.length != groundTruth.length) {
			throw new RuntimeException("The number of ground truths does not match the number of pixels");
		}

		// Create samples
		Sample[] samples = new Sample[groundTruth.length];
		for (int i = 0; i < pixelValuesForSelectedBands.length; i++) {
			samples[i] = new Sample(i, groundTruth[i], pixelValuesForSelectedBands[i]);
		}


		// Create classification result
		// Using numClasses - 1 because the background class is not included in the classification result
		ClassificationResult classificationResult = new ClassificationResult(numClasses - 1);


		// Run classification runs
		for (int i = 0; i < numClassificationRuns; i++) {


			// Shuffle and split into training and test set
			Sample[][] split = splitSamples(samples, trainingRatio);
			Sample[] trainingSamples = split[0];
			Sample[] testSamples = split[1];


			svm_model model = train(trainingSamples);

			List<Prediction> predictions = evaluateAccuracy(model, testSamples, numClasses);
			classificationResult.addRun(predictions);

		}


		return classificationResult;

	}

	/**
	 * Train both models with the same training data, and generate contingency table.
	 *
	 * @param selectedBands1
	 * @param selectedBands2
	 * @param trainingRatio
	 */
	public int[][] compareBandSubsets(List<Integer> selectedBands1, List<Integer> selectedBands2, double trainingRatio, boolean useGridSearch) {
		log.info("Comparing band subsets with selected bands: " + selectedBands1 + " and " + selectedBands2);


		// Load features and ground truth
		int[] groundTruth = dataset.getGroundTruthFlattenedAsArray();
		double[][] pixelValuesForSelectedBands1 = dataset.getBandsFlattened(selectedBands1).transpose().toDoubleMatrix();
		double[][] pixelValuesForSelectedBands2 = dataset.getBandsFlattened(selectedBands2).transpose().toDoubleMatrix();

		// Count number of classes
		int numClasses = Arrays.stream(groundTruth).max().getAsInt() + 1;

		// Create samples
		Sample[] samples1 = new Sample[groundTruth.length];
		Sample[] samples2 = new Sample[groundTruth.length];
		for (int i = 0; i < pixelValuesForSelectedBands1.length; i++) {
			samples1[i] = new Sample(i, groundTruth[i], pixelValuesForSelectedBands1[i]);
			samples2[i] = new Sample(i, groundTruth[i], pixelValuesForSelectedBands2[i]);
		}


		// Create classification result
		// Using numClasses - 1 because the background class is not included in the classification result
		ClassificationResult classificationResult1 = new ClassificationResult(numClasses - 1);
		ClassificationResult classificationResult2 = new ClassificationResult(numClasses - 1);


		// Shuffle and split into training and test set, collectively
		Sample[][][] split = splitSamplesForComparison(samples1, samples2, trainingRatio);

		Sample[] trainingSamples1 = split[0][0];
		Sample[] testSamples1 = split[0][1];
		Sample[] trainingSamples2 = split[1][0];
		Sample[] testSamples2 = split[1][1];


		svm_model model1 = null;
		svm_model model2 = null;

		if (useGridSearch) {
			model1 = train(trainingSamples1);
			model2 = train(trainingSamples2);
		} else {
			model1 = trainWithoutGridSearch(trainingSamples1);
			model2 = trainWithoutGridSearch(trainingSamples2);
		}
		List<Prediction> predictions1 = evaluateAccuracy(model1, testSamples1, numClasses);
		List<Prediction> predictions2 = evaluateAccuracy(model2, testSamples2, numClasses);
		classificationResult1.addRun(predictions1);
		classificationResult2.addRun(predictions2);

		// Generate contingency table
		int[][] contingencyTable = constructContingencyTable(predictions1, predictions2);

		return contingencyTable;


	}


	private svm_model train(Sample[] data) {


		log.info("Training SVM classifier with " + data.length + " samples");
		svm_problem trainingProblem = createProblem(data);

		// Find the best parameters using grid search
		svm_parameter bestParam = SVMParameterSearch.findBestParameters(trainingProblem);

		long startTime = System.currentTimeMillis();
		svm_model model = svm.svm_train(trainingProblem, bestParam);
		long endTime = System.currentTimeMillis();
		log.info("Training took " + (endTime - startTime) + " ms, excluding parameter search");

		return model;
	}

	private svm_model trainWithoutGridSearch(Sample[] data) {


		log.info("Training SVM classifier w.o. grid search with " + data.length + " samples");
		svm_problem trainingProblem = createProblem(data);

		// Find the best parameters using grid search
		svm_parameter param = new svm_parameter();
		param.svm_type = svm_parameter.C_SVC;
		param.kernel_type = svm_parameter.RBF;
		param.gamma = 1;
		param.C = 10;
		param.eps = 0.001;
		param.cache_size = 100;


		long startTime = System.currentTimeMillis();
		svm_model model = svm.svm_train(trainingProblem, param);
		long endTime = System.currentTimeMillis();
		log.info("Training took " + (endTime - startTime) + " ms");

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

	private static List<Prediction> evaluateAccuracy(svm_model model, Sample[] testSamples, int numClasses) {
		log.info("Evaluating accuracy of SVM classifier with " + testSamples.length + " samples");
		long startTime = System.currentTimeMillis();
		int numCorrectPredictions = 0;
		int[][] confusionMatrix = new int[numClasses][numClasses];


		List<Prediction> predictions = new ArrayList<>();

		for (Sample sample : testSamples) {
			int trueLabel = sample.label();
			double[] features = sample.features();
			int predictedLabel = predict(model, features);

			// Record true label and prediction
			Prediction prediction = new Prediction(sample.pixelIndex(), trueLabel, predictedLabel);
			predictions.add(prediction);

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
		return predictions;
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
