package no.haavardsjef.experiments.preliminary;

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
import java.util.Objects;

/**
 * The purpose of this experiment is to find the best hyperparameters for the subsequent experiments.
 */
public class HyperParamExperiment implements IExperiment {

	private final float[] W_RANGE = {0.4f, 0.6f, 0.8f, 0.9f};
	private final float[] C1_RANGE = {1.0f, 1.5f, 2.0f, 2.5f};
	private final float[] C2_RANGE = {1.0f, 1.5f, 2.0f, 2.5f};

	@Override
	public void runExperiment() throws IOException {


		// Initialize new MLflow client to connect to local MLflow server
		MLFlow mlFlow = new MLFlow();

		// Create a new experiment
		mlFlow.initializeExperiment("hyperparam-experiment");

		Dataset dataset = new Dataset(DatasetName.indian_pines);
		dataset.setupSuperpixelContainer();

		// Parameters that are the constant
		double fuzziness = 2.0;
		int numClassificationRuns = 5;
		Bounds bounds = dataset.getBounds();

		DistanceMeasure distanceMeasure = DistanceMeasure.SP_MEAN_EUCLIDEAN;
		IObjectiveFunction objectiveFunction = new FuzzyCMeans(dataset, fuzziness, distanceMeasure);
		for (int i = 2; i < 11; i += 2) {

			int numberOfBandsToSelect = i;
			// Start a new run

			for (float w : W_RANGE) {
				for (float c1 : C1_RANGE) {
					for (float c2 : C2_RANGE) {

						String runName = String.format("w=%s_c1=%s_c2=%s", w, c1, c2);
						mlFlow.startRun(runName);

						PSOParams params = new PSOParams(numberOfBandsToSelect);
						params.w = w;
						params.c1 = c1;
						params.c2 = c2;


						long startTime = System.currentTimeMillis();


						// PSO-FCM to select cluster centers
						SwarmPopulation swarmPopulation = new SwarmPopulation(params.numParticles, numberOfBandsToSelect, bounds, objectiveFunction);
						Particle solution = swarmPopulation.optimize(params.numIterations, params.w, params.c1, params.c2, false, true);
						List<Integer> clusterCentroids = solution.getDiscretePositionSorted();

						// Log parameters
						mlFlow.logParam("dataset", dataset.getDatasetName().toString());
						mlFlow.logParam("distanceMeasure", distanceMeasure.toString());
						mlFlow.logPSOParams(params);
						mlFlow.logParam("NumClassificationRuns", String.valueOf(numClassificationRuns));
						mlFlow.logParam("clusterCentroids", clusterCentroids.toString());
						mlFlow.logParam("numIterationsRan", String.valueOf(swarmPopulation.numIterationsRan));
						mlFlow.logParam("fuzziness", String.valueOf(fuzziness));

						float fitness = objectiveFunction.evaluate(clusterCentroids);


						// Log metrics
						mlFlow.logMetric("fitness", fitness);

						long endTime = System.currentTimeMillis();
						long duration = (endTime - startTime) / 1000;

						mlFlow.logMetric("duration", duration);

						// End run
						mlFlow.endRun();
					}
				}
			}
		}


	}

	public static void main(String[] args) {
		try {
			new HyperParamExperiment().runExperiment();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
