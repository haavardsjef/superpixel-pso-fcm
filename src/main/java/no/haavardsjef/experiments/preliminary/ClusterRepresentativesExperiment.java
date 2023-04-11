package no.haavardsjef.experiments.preliminary;

import no.haavardsjef.classification.SVMClassifier;
import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;
import no.haavardsjef.experiments.IExperiment;
import no.haavardsjef.experiments.MLFlow;
import no.haavardsjef.fcm.utility.ClusterRepresentatives;
import no.haavardsjef.fcm.FuzzyCMeans;
import no.haavardsjef.objectivefunctions.IObjectiveFunction;
import no.haavardsjef.pso.PSOParams;
import no.haavardsjef.pso.Particle;
import no.haavardsjef.pso.SwarmPopulation;
import no.haavardsjef.utility.Bounds;
import no.haavardsjef.utility.DistanceMeasure;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.io.IOException;
import java.util.List;

public class ClusterRepresentativesExperiment implements IExperiment {

	public void runExperiment() throws IOException {

		Dataset dataset = new Dataset(DatasetName.indian_pines);
		IObjectiveFunction objectiveFunction = new FuzzyCMeans(dataset, 2.0, DistanceMeasure.PIXEL_EUCLIDEAN);
		Bounds bounds = dataset.getBounds();

		// Initialize new MLflow client to connect to local MLflow server
		MLFlow mlFlow = new MLFlow();

		// Create a new experiment
		String experimentName = "cluster-representatives";
		mlFlow.initializeExperiment(experimentName);


		for (int i = 30; i < 100; i += 5) {

			long startTime = System.currentTimeMillis();
			int numberOfBandsToSelect = i;
			PSOParams params = new PSOParams(numberOfBandsToSelect);
			int numClassificationRuns = 10;

			// PSO-FCM to select cluster centers
			SwarmPopulation swarmPopulation = new SwarmPopulation(params.numParticles, numberOfBandsToSelect, bounds, objectiveFunction);
			Particle solution = swarmPopulation.optimize(params.numIterations, params.w, params.c1, params.c2, false, true);

			List<Integer> clusterCentroids = solution.getDiscretePositionSorted();
			long endTime = System.currentTimeMillis();
			long duration = (endTime - startTime) / 1000;

			// For each method of selecting cluster centers
			for (int method = 0; method < 3; method++) {
				// Start a new run
				String runName = "method_" + method + "_numBands_" + numberOfBandsToSelect;
				mlFlow.startRun(runName);

				mlFlow.logParam("method", String.valueOf(method));

				// Log parameters
				mlFlow.logPSOParams(params);
				mlFlow.logParam("clusterCentroids", clusterCentroids.toString());
				mlFlow.logParam("numClassificationRuns", String.valueOf(numClassificationRuns));
				mlFlow.logParam("optimizationDurationSeconds", String.valueOf(duration));


				// Select representative bands
				ClusterRepresentatives clusterRepresentatives = new ClusterRepresentatives(dataset);
				clusterRepresentatives.hardClusterBands(clusterCentroids);
				List<Integer> selectedBands = null;
				if (method == 0) {
					selectedBands = clusterRepresentatives.centroidRepresentatives(clusterCentroids);
				} else if (method == 1) {
					selectedBands = clusterRepresentatives.meanRepresentative(clusterCentroids);
				} else if (method == 2) {
					selectedBands = clusterRepresentatives.highestEntropyRepresentative(clusterCentroids);
				}
				mlFlow.logParam("selectedBands", selectedBands.toString());

				// Log metrics
				SVMClassifier svmClassifier = new SVMClassifier(dataset);
				DescriptiveStatistics stats = svmClassifier.evaluate(selectedBands, numClassificationRuns);
				mlFlow.logMetric("accuracy", stats.getMean());
				mlFlow.logMetric("std", stats.getStandardDeviation());

				// End run
				mlFlow.endRun();
			}


		}
	}

	public static void main(String[] args) {

		ClusterRepresentativesExperiment experiment = new ClusterRepresentativesExperiment();
		try {
			experiment.runExperiment();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
}
