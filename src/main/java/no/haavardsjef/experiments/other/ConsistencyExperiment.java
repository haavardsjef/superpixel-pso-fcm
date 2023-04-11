package no.haavardsjef.experiments.other;

import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;
import no.haavardsjef.experiments.IExperiment;
import no.haavardsjef.experiments.MLFlow;
import no.haavardsjef.fcm.FuzzyCMeans;
import no.haavardsjef.objectivefunctions.IObjectiveFunction;
import no.haavardsjef.pso.PSOParams;
import no.haavardsjef.pso.Particle;
import no.haavardsjef.pso.SwarmPopulation;
import no.haavardsjef.utility.Bounds;

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
			PSOParams params = new PSOParams(numberOfBandsToSelect);

			long startTime = System.currentTimeMillis();


			// PSO-FCM to select cluster centers
			SwarmPopulation swarmPopulation = new SwarmPopulation(params.numParticles, numberOfBandsToSelect, bounds, objectiveFunction);
			Particle solution = swarmPopulation.optimize(params.numIterations, params.w, params.c1, params.c2, false, true);

			long endTime = System.currentTimeMillis();
			long duration = (endTime - startTime) / 1000;


			List<Integer> clusterCentroids = solution.getDiscretePositionSorted();

			String runName = "run_" + i;
			mlFlow.startRun(runName);

			// Log PSO params
			mlFlow.logPSOParams(params);

			mlFlow.logParam("optimizationTimeSeconds", String.valueOf(duration));
			mlFlow.logParam("distanceMetric", "pixel-wise-euclidean");
			mlFlow.logParam("fuzzines", String.valueOf(fuzziness));

			// Log parameters
			mlFlow.logParam("clusterCentroids", clusterCentroids.toString());
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
