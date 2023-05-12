package no.haavardsjef.experiments.plan;

import lombok.extern.log4j.Log4j2;
import no.haavardsjef.classification.SVMClassifier;
import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;
import no.haavardsjef.experiments.IExperiment;
import no.haavardsjef.experiments.preliminary.BaseLineExperiment;
import no.haavardsjef.utility.Bounds;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

@Log4j2
public class TrainingTimeExperiment implements IExperiment {
	@Override
	public void runExperiment() throws IOException {
		Dataset dataset = new Dataset(DatasetName.Salinas);
		Bounds b = dataset.getBounds();


		// Select all bands
//		List<Integer> selectedBands = IntStream.range(b.lower(), b.upper()).collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
		List<Integer> selectedBands = Arrays.asList(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19);

		SVMClassifier svmClassifier = new SVMClassifier(dataset);
		long startTime = System.currentTimeMillis();
		svmClassifier.justTrain(selectedBands, 10, 0.1);
		long endTime = System.currentTimeMillis();
		log.info("Time elapsed: " + (endTime - startTime) + "ms");


	}

	public static void main(String[] args) {
		BaseLineExperiment baseLineExperiment = new BaseLineExperiment();
		try {
			baseLineExperiment.runExperiment();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
