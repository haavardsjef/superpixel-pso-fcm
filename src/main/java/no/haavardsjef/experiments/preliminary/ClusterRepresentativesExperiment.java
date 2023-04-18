package no.haavardsjef.experiments.preliminary;

import no.haavardsjef.classification.ClassificationResult;
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
		dataset.setupSuperpixelContainer();
		DistanceMeasure distanceMeasure = DistanceMeasure.SP_MEAN_EUCLIDEAN;
		double fuzziness = 2.0;
		IObjectiveFunction objectiveFunction = new FuzzyCMeans(dataset, fuzziness, distanceMeasure);
		Bounds bounds = dataset.getBounds();

		// Initialize new MLflow client to connect to local MLflow server
		String trackingUri = "http://35.185.118.215:8080";
		MLFlow mlFlow = new MLFlow(trackingUri);

		// Create a new experiment
		String experimentName = "cluster-representatives";
		mlFlow.initializeExperiment(experimentName);

		for (int r = 0; r < 11; r++) {

			for (int i = 5; i < 50; i += 5) {

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
					String runName = "method_" + method + "_numBands_" + numberOfBandsToSelect + "-" + r;
					mlFlow.startRun(runName);

					mlFlow.logParam("method", String.valueOf(method));
					mlFlow.logParam("corrected", "True");
					mlFlow.logParam("dataset", dataset.getDatasetName().toString());
					mlFlow.logParam("distanceMeasure", distanceMeasure.toString());
					mlFlow.logParam("fuzziness", String.valueOf(fuzziness));
					mlFlow.logParam("numBands", String.valueOf(numberOfBandsToSelect));
					mlFlow.logParam("numIterationsRan", String.valueOf(swarmPopulation.numIterationsRan));


					// Log parameters
					mlFlow.logPSOParams(params);
					mlFlow.logParam("clusterCentroids", clusterCentroids.toString());
					mlFlow.logParam("optimizationDurationSeconds", String.valueOf(duration));


					// Select representative bands
					ClusterRepresentatives clusterRepresentatives = new ClusterRepresentatives(dataset);
					clusterRepresentatives.hardClusterBands(clusterCentroids);
					List<Integer> selectedBands = null;
					if (method == 0) {
						selectedBands = clusterRepresentatives.centroidRepresentatives(clusterCentroids);
						mlFlow.logParam("CRMethod", "centroid");
					} else if (method == 1) {
						selectedBands = clusterRepresentatives.meanRepresentative(clusterCentroids);
						mlFlow.logParam("CRMethod", "mean");
					} else if (method == 2) {
						selectedBands = clusterRepresentatives.highestEntropyRepresentative(clusterCentroids);
						mlFlow.logParam("CRMethod", "entropy");
					}
					mlFlow.logParam("selectedBands", selectedBands.toString());

					// Classification
					SVMClassifier svm = new SVMClassifier(dataset);
					ClassificationResult result = svm.evaluate(selectedBands, numClassificationRuns);
					DescriptiveStatistics OO = result.getOverallAccuracy();
					DescriptiveStatistics AOA = result.getAverageOverallAccuracy();
					mlFlow.logParam("numClassificationRuns", String.valueOf(numClassificationRuns));

					// Log metrics
					mlFlow.logMetric("OA_mean", OO.getMean());
					mlFlow.logMetric("OA_std", OO.getStandardDeviation());
					mlFlow.logMetric("AOA_mean", AOA.getMean());
					mlFlow.logMetric("AOA_std", AOA.getStandardDeviation());

					// End run
					mlFlow.endRun();
				}

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
