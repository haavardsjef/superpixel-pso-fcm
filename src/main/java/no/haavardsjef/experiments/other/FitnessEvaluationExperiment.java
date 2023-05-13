package no.haavardsjef.experiments.other;

import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;
import no.haavardsjef.experiments.IExperiment;
import no.haavardsjef.fcm.FuzzyCMeans;
import no.haavardsjef.utility.DistanceMeasure;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class FitnessEvaluationExperiment implements IExperiment {
	@Override
	public void runExperiment() throws IOException {
		Dataset ds = new Dataset(DatasetName.indian_pines, true);
		ds.setupSuperpixelContainer(400, 1000f);
		ds.calculateKlDivergencesSuperpixelLevel();

		FuzzyCMeans fcm = new FuzzyCMeans(ds, 2.0, DistanceMeasure.SP_LEVEL_KL_DIVERGENCE_L1NORM);

		List<Integer> selectedBands = Arrays.asList(63, 96, 113, 116, 119, 125, 129, 133, 136, 177);

		float fitness = fcm.evaluate(selectedBands);

		System.out.println("Fitness: " + fitness);

	}

	public static void main(String[] args) {
		FitnessEvaluationExperiment experiment = new FitnessEvaluationExperiment();
		try {
			experiment.runExperiment();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
