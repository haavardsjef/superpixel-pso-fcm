package no.haavardsjef.experiments;

import no.haavardsjef.classification.SVMClassifier;
import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;
import no.haavardsjef.fcm.ClusterRepresentatives;
import no.haavardsjef.fcm.FuzzyCMeans;
import no.haavardsjef.objectivefunctions.IObjectiveFunction;
import no.haavardsjef.pso.Particle;
import no.haavardsjef.pso.SwarmPopulation;
import no.haavardsjef.utility.Bounds;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.io.IOException;
import java.util.List;

/**
 * Evaluate the consistence of the BS - method
 */
public class ConsistencyExperiment implements IExperiment {
	@Override
	public void runExperiment() throws IOException {
		Dataset dataset = new Dataset(DatasetName.indian_pines);
		double fuzziness = 2.0;
		IObjectiveFunction objectiveFunction = new FuzzyCMeans(dataset, fuzziness);
		Bounds bounds = dataset.getBounds();

		// Initialize new MLflow client to connect to local MLflow server
		MLFlow mlFlow = new MLFlow();

		// Create a new experiment
		String experimentName = "consistency-experiment";
		mlFlow.initializeExperiment(experimentName);


		for (int i = 1; i < 10; i += 1) {

			int numberOfBandsToSelect = 2;
			int numParticles = 100;
			int numIterations = 100;
			float w = 0.5f;
			float c1 = 0.5f;
			float c2 = 0.2f;


			long startTime = System.currentTimeMillis();

			// PSO-FCM to select cluster centers
			SwarmPopulation swarmPopulation = new SwarmPopulation(numParticles, numberOfBandsToSelect, bounds, objectiveFunction);
			Particle solution = swarmPopulation.optimize(numIterations, w, c1, c2, false);

			long endTime = System.currentTimeMillis();
			long duration = (endTime - startTime) / 1000;


			List<Integer> clusterCentroids = solution.getDiscretePositionSorted();

			String runName = "run_" + i;
			mlFlow.startRun(runName);

			mlFlow.logParam("optimizationTimeSeconds", String.valueOf(duration));
			mlFlow.logParam("distanceMetric", "pixel-wise-euclidean");
			mlFlow.logParam("fuzzines", String.valueOf(fuzziness));
			mlFlow.logParam("numBands", String.valueOf(numberOfBandsToSelect));

			// Log parameters
			mlFlow.logParam("clusterCentroids", clusterCentroids.toString());
			mlFlow.logParam("numParticles", String.valueOf(numParticles));
			mlFlow.logParam("numIterations", String.valueOf(numIterations));
			mlFlow.logParam("w", String.valueOf(w));
			mlFlow.logParam("c1", String.valueOf(c1));
			mlFlow.logParam("c2", String.valueOf(c2));
			mlFlow.logParam("datasetName", dataset.getDatasetName().toString());
			mlFlow.logMetric("fitness", solution.evaluate());
			mlFlow.endRun();


		}


	}

	public static void main(String[] args) {
		ConsistencyExperiment consistencyExperiment = new ConsistencyExperiment();
		try {
			consistencyExperiment.runExperiment();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
