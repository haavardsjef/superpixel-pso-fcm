package no.haavardsjef.classification;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang3.ArrayUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.*;
import java.util.function.Function;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toMap;


import static java.util.stream.Collectors.groupingBy;

@Log4j2
public class ClassificationUtilities {

	public static Sample[][] splitSamples(Sample[] samples, double trainingRatio) {
		// Create empty training and testing set
		List<Sample> trainingSamples = new ArrayList<>();
		List<Sample> testSamples = new ArrayList<>();


		// Remove unlabeled pixels/samples
		Sample[] filtered = Arrays.stream(samples).filter(s -> s.label() != 0).toArray(Sample[]::new);

		// Group samples by label
		Map<Integer, List<Sample>> grouped = Arrays.stream(filtered).collect(groupingBy(Sample::label));

		// For each label, shuffle and then select training and testing samples
		grouped.entrySet().forEach(sampleList -> {
			Collections.shuffle(sampleList.getValue());
			long splitIndex = (long) Math.ceil(sampleList.getValue().size() * trainingRatio);
			trainingSamples.addAll(sampleList.getValue().subList(0, (int) splitIndex));
			testSamples.addAll(sampleList.getValue().subList((int) splitIndex, sampleList.getValue().size()));
		});

		// Convert training and test samples lists to arrays
		Sample[] trainingSamplesArray = trainingSamples.toArray(new Sample[0]);
		Sample[] testSamplesArray = testSamples.toArray(new Sample[0]);

		// Normalize samples together
		Sample[][] normalized = normalizeTogether(trainingSamplesArray, testSamplesArray);

		log.info("Split samples into " + trainingSamplesArray.length + " training samples and " + testSamplesArray.length + " test samples.");

		return normalized;
	}

	/**
	 * Same as method above, but makes sure that both Sample[] are split in the same way.
	 */
	public static Sample[][][] splitSamplesForComparison(Sample[] samples1, Sample[] samples2, double trainingRatio) {
		// Create empty training and testing set
		List<Sample> trainingSamples = new ArrayList<>();
		List<Sample> testSamples = new ArrayList<>();


		// Remove unlabeled pixels/samples
		Sample[] filtered = Arrays.stream(samples1).filter(s -> s.label() != 0).toArray(Sample[]::new);

		// Group samples by label
		Map<Integer, List<Sample>> grouped = Arrays.stream(filtered).collect(groupingBy(Sample::label));

		// For each label, shuffle and then select training and testing samples
		grouped.entrySet().forEach(sampleList -> {
			Collections.shuffle(sampleList.getValue());
			long splitIndex = (long) Math.ceil(sampleList.getValue().size() * trainingRatio);
			trainingSamples.addAll(Arrays.asList(Arrays.copyOfRange(sampleList.getValue().toArray(new Sample[0]), 0, (int) splitIndex)));
			testSamples.addAll(Arrays.asList(Arrays.copyOfRange(sampleList.getValue().toArray(new Sample[0]), (int) splitIndex, sampleList.getValue().size())));
		});

		// Create map of pixel indices to samples in samples2
		Map<Integer, Sample> samples2Map = Arrays.stream(samples2).collect(toMap(Sample::pixelIndex, Function.identity()));

		// Split samples2 in the same way as samples1
		List<Sample> trainingSamples2 = new ArrayList<>();
		List<Sample> testSamples2 = new ArrayList<>();
		for (Sample trainingSample : trainingSamples) {
			Sample sample2 = samples2Map.get(trainingSample.pixelIndex());
			if (sample2 != null) {
				trainingSamples2.add(sample2);
			}
		}
		for (Sample testSample : testSamples) {
			Sample sample2 = samples2Map.get(testSample.pixelIndex());
			if (sample2 != null) {
				testSamples2.add(sample2);
			}
		}

		// Verify that index 0 in both lists have the same pixel index
		if (trainingSamples.get(0).pixelIndex() != trainingSamples2.get(0).pixelIndex()) {
			throw new RuntimeException("Pixel index mismatch");
		}

		// Convert training and test samples lists to arrays
		Sample[] trainingSamplesArray1 = trainingSamples.toArray(new Sample[0]);
		Sample[] testSamplesArray1 = testSamples.toArray(new Sample[0]);
		Sample[] trainingSamplesArray2 = trainingSamples2.toArray(new Sample[0]);
		Sample[] testSamplesArray2 = testSamples2.toArray(new Sample[0]);

		// Normalize samples together
		Sample[][] normalized1 = normalizeTogether(trainingSamplesArray1, testSamplesArray1);
		Sample[][] normalized2 = normalizeTogether(trainingSamplesArray2, testSamplesArray2);

		return new Sample[][][]{normalized1, normalized2};
	}

	public static Sample[][] normalizeTogether(Sample[] training, Sample[] testing) {
		Sample[] all = ArrayUtils.addAll(training, testing);

		double[] featureMin = IntStream.range(0, all[0].features().length)
				.mapToDouble(b -> Arrays.stream(all).mapToDouble(s -> s.features()[b]).min().getAsDouble())
				.toArray();
		double[] featureMax = IntStream.range(0, all[0].features().length)
				.mapToDouble(b -> Arrays.stream(all).mapToDouble(s -> s.features()[b]).max().getAsDouble())
				.toArray();

		// Normalize training samples
		for (Sample sample : training) {
			double[] features = sample.features();
			for (int i = 0; i < features.length; i++) {
				features[i] = (features[i] - featureMin[i]) / (featureMax[i] - featureMin[i]);
			}
		}

		// Normalize test samples
		for (Sample sample : testing) {
			double[] features = sample.features();
			for (int i = 0; i < features.length; i++) {
				features[i] = (features[i] - featureMin[i]) / (featureMax[i] - featureMin[i]);
			}
		}

		return new Sample[][]{training, testing};
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
				writer.println(row);
			}
		} catch (FileNotFoundException e) {
			System.err.println("Error saving confusion matrix to CSV file: " + e.getMessage());
		}
	}

	public static int[][] constructContingencyTable(List<Prediction> predictions1, List<Prediction> predictions2) {
		// Verify that both lists have the same size
		if (predictions1.size() != predictions2.size()) {
			throw new RuntimeException("Prediction lists have different sizes");
		}

		// Create contingency table
		int[][] contingencyTable = new int[2][2];

		// For each prediction, increment the corresponding cell in the contingency table
		for (int i = 0; i < predictions1.size(); i++) {
			Prediction prediction1 = predictions1.get(i);
			Prediction prediction2 = predictions2.get(i);

			// Make sure the pixelIndex is the same for both predictions
			if (prediction1.pixelIndex() != prediction2.pixelIndex()) {
				throw new RuntimeException("Pixel index mismatch");
			}


			if (prediction1.trueLabel() == prediction1.predictedLabel() && prediction2.trueLabel() == prediction2.predictedLabel()) {
				contingencyTable[0][0]++;
			} else if (prediction1.trueLabel() == prediction1.predictedLabel() && prediction2.trueLabel() != prediction2.predictedLabel()) {
				contingencyTable[0][1]++;
			} else if (prediction1.trueLabel() != prediction1.predictedLabel() && prediction2.trueLabel() == prediction2.predictedLabel()) {
				contingencyTable[1][0]++;
			} else if (prediction1.trueLabel() != prediction1.predictedLabel() && prediction2.trueLabel() != prediction2.predictedLabel()) {
				contingencyTable[1][1]++;
			}
		}
		return contingencyTable;
	}
}
