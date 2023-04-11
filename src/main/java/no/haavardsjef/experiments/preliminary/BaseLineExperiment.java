package no.haavardsjef.experiments.preliminary;

import lombok.extern.log4j.Log4j2;
import no.haavardsjef.classification.SVMClassifier;
import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;
import no.haavardsjef.experiments.IExperiment;
import no.haavardsjef.utility.Bounds;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

@Log4j2
public class BaseLineExperiment implements IExperiment {
	@Override
	public void runExperiment() throws IOException {
		Dataset dataset = new Dataset(DatasetName.indian_pines);
		Bounds b = dataset.getBounds();


		// Select all bands
		List<Integer> selectedBands = IntStream.range(b.lower(), b.upper()).collect(ArrayList::new, ArrayList::add, ArrayList::addAll);

		SVMClassifier svmClassifier = new SVMClassifier(dataset);
		DescriptiveStatistics stats = svmClassifier.evaluate(selectedBands, 20);

		log.info("Mean: " + stats.getMean());
		log.info("Standard deviation: " + stats.getStandardDeviation());

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