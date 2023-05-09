package no.haavardsjef.experiments.plan;

import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;
import no.haavardsjef.experiments.IExperiment;
import no.haavardsjef.utility.DistanceMeasure;

import java.io.IOException;

public class DistanceMetricRuntimeExperiment implements IExperiment {
	@Override
	public void runExperiment() throws IOException {

		Dataset ds = new Dataset(DatasetName.indian_pines, true);

		// Time how long to setup
		long startTime = System.nanoTime();
		ds.setupSuperpixelContainer(400, 1000f);
//		ds.calculateProbabilityDistributionsSPmean(); //SP_MEAN_KL_Divergence, SP_MEAN_DISJOINT, SP_MEAN_COR_COF
//		ds.calculateDisjointInfoSuperpixelLevel(); //SP_MEAN_DISJOINT
//		ds.calculateKlDivergencesSuperpixelLevel(); //SP_LEVEL_KL_DIVERGENCE_L1NORM}


		long endTime = System.nanoTime();
		long duration = (endTime - startTime);
		System.out.println("Setup time: " + duration / 1000000 + "ms");

		// Calculate 1,000,000 distances
		long distanceStartTime = System.nanoTime();
		for (int i = 0; i < 100; i++) {
			ds.distance(DistanceMeasure.SP_MEAN_EUCLIDEAN, 0, 1);
		}
		endTime = System.nanoTime();
		duration = (endTime - distanceStartTime);
		System.out.println("Time to calculate 1,000,000 distances: " + duration / 1000000 + "ms");


		// Total time
		endTime = System.nanoTime();
		duration = (endTime - startTime);
		System.out.println("Total time: " + duration / 1000000 + "ms");


	}

	public static void main(String[] args) {
		DistanceMetricRuntimeExperiment experiment = new DistanceMetricRuntimeExperiment();
		try {
			experiment.runExperiment();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
