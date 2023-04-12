package no.haavardsjef.classification;

import libsvm.*;
import lombok.extern.log4j.Log4j2;
import no.haavardsjef.dataset.Dataset;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static no.haavardsjef.classification.ClassificationUtilities.saveConfusionMatrixToCSV;
import static no.haavardsjef.classification.ClassificationUtilities.splitSamples;

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


		ClassificationResult classificationResult = new ClassificationResult();


		// Run classification runs
		for (int i = 0; i < numClassificationRuns; i++) {


			// Shuffle and split into training and test set
			double trainingRatio = 0.1;
			Sample[][] split = splitSamples(samples, trainingRatio);
			Sample[] trainingSamples = split[0];
			Sample[] testSamples = split[1];


			svm_model model = train(trainingSamples);

			List<Prediction> predictions = evaluateAccuracy(model, testSamples, numClasses);
			classificationResult.addRun(predictions);

		}


		return classificationResult;

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
