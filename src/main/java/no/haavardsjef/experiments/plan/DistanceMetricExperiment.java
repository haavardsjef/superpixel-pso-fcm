package no.haavardsjef.experiments.plan;

import lombok.extern.log4j.Log4j2;
import no.haavardsjef.classification.ClassificationResult;
import no.haavardsjef.classification.SVMClassifier;
import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;
import no.haavardsjef.experiments.IExperiment;
import no.haavardsjef.experiments.MLFlow;
import no.haavardsjef.fcm.FuzzyCMeans;
import no.haavardsjef.fcm.utility.ClusterRepresentatives;
import no.haavardsjef.objectivefunctions.IObjectiveFunction;
import no.haavardsjef.pso.PSOParams;
import no.haavardsjef.pso.Particle;
import no.haavardsjef.pso.SwarmPopulation;
import no.haavardsjef.utility.Bounds;
import no.haavardsjef.utility.DistanceMeasure;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.io.IOException;
import java.util.List;

@Log4j2
public class DistanceMetricExperiment implements IExperiment {


	public void runExperiment() throws IOException {
		// Initialize new MLflow client to connect to local MLflow server
		MLFlow mlFlow = new MLFlow();

		// Create a new experiment
		String experimentName = "distance-metric-experiment";
		mlFlow.initializeExperiment(experimentName);


		Dataset dataset = new Dataset(DatasetName.indian_pines);
		dataset.setupSuperpixelContainer();

		// Parameters that are the constant
		double fuzziness = 2.0;
		int numClassificationRuns = 10;
		Bounds bounds = dataset.getBounds();

		// For every distance measure
		for (DistanceMeasure distanceMeasure : DistanceMeasure.values()) {

			IObjectiveFunction objectiveFunction = new FuzzyCMeans(dataset, fuzziness, distanceMeasure);
			for (int i = 1; i < 30; i += 2) {

				int numberOfBandsToSelect = i;
				// Start a new run
				String runName = distanceMeasure + "_numBands_" + i;
				mlFlow.startRun(runName);


				long startTime = System.currentTimeMillis();
				PSOParams params = new PSOParams(numberOfBandsToSelect);
				params.w = 0.75f;
				params.c1 = 1.0f;
				params.c2 = 1.0f;


				// PSO-FCM to select cluster centers
				SwarmPopulation swarmPopulation = new SwarmPopulation(params.numParticles, numberOfBandsToSelect, bounds, objectiveFunction);
				Particle solution = swarmPopulation.optimize(params.numIterations, params.w, params.c1, params.c2, false, true);
				List<Integer> clusterCentroids = solution.getDiscretePositionSorted();

				// Log parameters
				mlFlow.logParam("distanceMeasure", distanceMeasure.toString());
				mlFlow.logPSOParams(params);
				mlFlow.logParam("NumClassificationRuns", String.valueOf(numClassificationRuns));
				mlFlow.logParam("clusterCentroids", clusterCentroids.toString());
				mlFlow.logParam("numIterationsRan", String.valueOf(swarmPopulation.numIterationsRan);

				ClusterRepresentatives cr = new ClusterRepresentatives(dataset);
				cr.hardClusterBands(clusterCentroids);

				List<Integer> selectedBands = cr.meanRepresentative(clusterCentroids);
				mlFlow.logParam("selectedBands", selectedBands.toString());
				mlFlow.logParam("CRMethod", "meanRepresentative");

				// Evaluate using SVMClassifier
				SVMClassifier svmClassifier = new SVMClassifier(dataset);
				ClassificationResult result = svmClassifier.evaluate(selectedBands, numClassificationRuns);
				DescriptiveStatistics OA = result.getOverallAccuracy();
				DescriptiveStatistics AOA = result.getAverageOverallAccuracy();

				// Log metrics
				mlFlow.logMetric("OA_mean", OA.getMean());
				mlFlow.logMetric("OA_std", OA.getStandardDeviation());
				mlFlow.logMetric("AOA_mean", AOA.getMean());
				mlFlow.logMetric("AOA_std", AOA.getStandardDeviation());

				long endTime = System.currentTimeMillis();
				long duration = (endTime - startTime) / 1000;

				mlFlow.logMetric("duration", duration);

				// End run
				mlFlow.endRun();
			}

		}


	}

	public static void main(String[] args) throws IOException {
		DistanceMetricExperiment distanceMetricExperiment = new DistanceMetricExperiment();
		distanceMetricExperiment.runExperiment();
	}
}
