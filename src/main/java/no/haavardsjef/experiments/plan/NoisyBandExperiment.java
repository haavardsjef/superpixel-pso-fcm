package no.haavardsjef.experiments.plan;

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

public class NoisyBandExperiment implements IExperiment {
	@Override
	public void runExperiment() throws IOException {

		Dataset dataset = new Dataset(DatasetName.indian_pines);
		dataset.setupSuperpixelContainer();
		Bounds bounds = dataset.getBounds();
		DistanceMeasure distanceMeasure = DistanceMeasure.SP_MEAN_EUCLIDEAN;

		double fuzziness = 2;
		IObjectiveFunction objectiveFunction = new FuzzyCMeans(dataset, 2.0, distanceMeasure);

		// Initialize new MLflow client to connect to local MLflow server
		MLFlow mlFlow = new MLFlow();

		// Create a new experiment
		String experimentName = "noisy-band-experiment";
		mlFlow.initializeExperiment(experimentName);

		long startTime = System.currentTimeMillis();

		for (int i = 4; i < 31; i += 2) {

			int numberOfBandsToSelect = i;


			PSOParams params = new PSOParams(numberOfBandsToSelect);
			int numClassificationRuns = 10;

			// Start run and log params
			mlFlow.startRun("corrected");
			mlFlow.logParam("corrected", "True");
			mlFlow.logParam("fuzziness", String.valueOf(fuzziness));
			mlFlow.logPSOParams(params);
			mlFlow.logParam("dataset", dataset.getDatasetName().toString());
			mlFlow.logParam("distanceMeasure", distanceMeasure.toString());

			// PSO-FCM to select cluster centers
			SwarmPopulation swarmPopulation = new SwarmPopulation(params.numParticles, numberOfBandsToSelect, bounds, objectiveFunction);
			Particle solution = swarmPopulation.optimize(params.numIterations, params.w, params.c1, params.c2, false, true);
			List<Integer> clusterCenters = solution.getDiscretePositionSorted();
			mlFlow.logParam("clusterCenters", clusterCenters.toString());
			mlFlow.logParam("numIterationsRan", String.valueOf(swarmPopulation.numIterationsRan));

			ClusterRepresentatives cr = new ClusterRepresentatives(dataset);
			cr.hardClusterBands(clusterCenters);
			List<Integer> selectedBands = cr.highestEntropyRepresentative(clusterCenters);
			mlFlow.logParam("selectedBands", selectedBands.toString());
			mlFlow.logParam("CRMethod", "highest-entropy");


			// Classification
			SVMClassifier svm = new SVMClassifier(dataset);
			ClassificationResult result = svm.evaluate(selectedBands, numClassificationRuns);
			DescriptiveStatistics OO = result.getOverallAccuracy();
			DescriptiveStatistics AOA = result.getAverageOverallAccuracy();
			mlFlow.logParam("numClassificationRuns", String.valueOf(numClassificationRuns));

			// Log metrics
			mlFlow.logMetric("OO_mean", OO.getMean());
			mlFlow.logMetric("OO_std", OO.getStandardDeviation());
			mlFlow.logMetric("AOA_mean", AOA.getMean());
			mlFlow.logMetric("AOA_std", AOA.getStandardDeviation());


			// End run
			mlFlow.endRun();

		}

	}

	public static void main(String[] args) {
		try {
			NoisyBandExperiment experiment = new NoisyBandExperiment();
			experiment.runExperiment();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
