package no.haavardsjef.experiments.preliminary;

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
public class SuperpixelParamsExperiment implements IExperiment {
	@Override
	public void runExperiment() throws IOException {

		int[] numSuperparticlesRange = {100, 200, 300};
		float[] spatialWeightsRange = {1000f, 10000f, 100000f, 1000000f, 10000000f};


		// Initialize new MLflow client to connect to local MLflow server
		MLFlow mlFlow = new MLFlow("http://35.185.118.215:8080/");

		// Create a new experiment
		String experimentName = "superpixel-params-experiment";
		mlFlow.initializeExperiment(experimentName);


		// Iterate numSuperpixelsRange
		for (int numSuperpixels : numSuperparticlesRange) {

			// Iterate spatialWeightsRange
			for (float spatialWeight : spatialWeightsRange) {


				Dataset dataset = new Dataset(DatasetName.indian_pines); // Choose dataset
				dataset.setupSuperpixelContainer(numSuperpixels, spatialWeight); // Setup superpixel container, needs to be done if using superpixels
				DistanceMeasure distanceMeasure = DistanceMeasure.SP_MEAN_EUCLIDEAN; // Choose distance measure
				Bounds bounds = dataset.getBounds(); // Get bounds for PSO
				IObjectiveFunction fcm = new FuzzyCMeans(dataset, 2.0, distanceMeasure);


				int numberOfBandsToSelect = 10;
				PSOParams params = new PSOParams(numberOfBandsToSelect); // Using default pso parameters

				for (int run = 0; run < 5; run++) {
					// Start new run
					mlFlow.startRun("run-" + run + "-numSuperpixels-" + numSuperpixels + "-spatialWeight-" + spatialWeight);
					mlFlow.logParam("numSuperpixels", String.valueOf(numSuperpixels));
					mlFlow.logParam("spatialWeight", String.valueOf(spatialWeight));
					mlFlow.logParam("dataset", dataset.getDatasetName().toString());
					mlFlow.logParam("distanceMeasure", distanceMeasure.toString());
					mlFlow.logParam("numberOfBandsToSelect", String.valueOf(numberOfBandsToSelect));
					mlFlow.logPSOParams(params);

					log.info("Evaluating run " + run + " with numSuperpixels " + numSuperpixels + " and spatialWeight " + spatialWeight + " on dataset " + dataset.getDatasetName().toString());

					// PSO-FCM to select cluster centers
					SwarmPopulation swarmPopulation = new SwarmPopulation(params.numParticles, numberOfBandsToSelect, bounds, fcm);
					Particle solution = swarmPopulation.optimize(params.numIterations, params.w, params.c1, params.c2, false, true);
					List<Integer> clusterCenters = solution.getDiscretePositionSorted();

					// Select cluster representatives
					ClusterRepresentatives cr = new ClusterRepresentatives(dataset);
					cr.hardClusterBands(clusterCenters);
					List<Integer> selectedBands = cr.highestEntropyRepresentative(clusterCenters); // In this example, select with highest entropy

					// Log selected bands
					mlFlow.logParam("clusterCenters", clusterCenters.toString());
					mlFlow.logParam("selectedBands", selectedBands.toString());

					// SVM Classification
					int numClassificationRuns = 10;
					SVMClassifier classifier = new SVMClassifier(dataset);
					ClassificationResult result = classifier.evaluate(selectedBands, numClassificationRuns);
					mlFlow.logParam("numClassificationRuns", String.valueOf(numClassificationRuns));

					DescriptiveStatistics OA = result.getOverallAccuracy();
					DescriptiveStatistics AOA = result.getAverageOverallAccuracy();

					// Log results
					mlFlow.logMetric("OA", OA.getMean());
					mlFlow.logMetric("OA_SD", OA.getStandardDeviation());
					mlFlow.logMetric("AOA", AOA.getMean());
					mlFlow.logMetric("AOA_SD", AOA.getStandardDeviation());

					log.info("OA: " + OA.getMean() + " ( SD:" + OA.getStandardDeviation() + ")");
					log.info("AOA: " + AOA.getMean() + " ( SD:" + AOA.getStandardDeviation() + ")");
					mlFlow.endRun();
				}
			}
		}
	}

	public static void main(String[] args) {
		try {
			new SuperpixelParamsExperiment().runExperiment();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}


}
