package no.haavardsjef.classification;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang3.ArrayUtils;

import java.util.*;
import java.util.stream.IntStream;

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
}
