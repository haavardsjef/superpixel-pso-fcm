package no.haavardsjef.experiments.other;

import lombok.extern.log4j.Log4j2;
import no.haavardsjef.classification.ClassificationResult;
import no.haavardsjef.classification.SVMClassifier;
import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;
import no.haavardsjef.experiments.IExperiment;
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
public class ExampleExperiment implements IExperiment {
	@Override
	public void runExperiment() throws IOException {
		Dataset dataset = new Dataset(DatasetName.Pavia); // Choose dataset
//		dataset.precomputeEuclideanDistances();
		dataset.setupSuperpixelContainer(900, 10000f); // Setup superpixel container, needs to be done if using superpixels
//		dataset.calculateProbabilityDistributionsSPmean(); //SP_MEAN_KL_Divergence, SP_MEAN_DISJOINT, SP_MEAN_COR_COF
//		dataset.calculateDisjointInfoSuperpixelLevel(); //SP_MEAN_DISJOINT
		dataset.calculateKlDivergencesSuperpixelLevel(); //SP_LEVEL_KL_DIVERGENCE_L1NORM}

		DistanceMeasure distanceMeasure = DistanceMeasure.SP_LEVEL_KL_DIVERGENCE_L1NORM; // Choose distance measure
		Bounds bounds = dataset.getBounds(); // Get bounds for PSO
		IObjectiveFunction fcm = new FuzzyCMeans(dataset, 2.0, distanceMeasure);

		int numberOfBandsToSelect = 20;
		PSOParams params = new PSOParams(numberOfBandsToSelect); // Using default pso parameters

		long startTime = System.currentTimeMillis();
		// PSO-FCM to select cluster centers
		SwarmPopulation swarmPopulation = new SwarmPopulation(params.numParticles, numberOfBandsToSelect, bounds, fcm);
		Particle solution = swarmPopulation.optimize(100, params.w, params.c1, params.c2, false, false);
		long endTime = System.currentTimeMillis();
		long duration = (endTime - startTime);
		log.info("PSO-FCM time: " + duration + "ms");

		List<Integer> clusterCenters = solution.getDiscretePositionSorted();

		// Select cluster representatives
		ClusterRepresentatives cr = new ClusterRepresentatives(dataset);
		cr.hardClusterBands(clusterCenters);
		List<Integer> selectedBands = cr.highestEntropyRepresentative(clusterCenters); // In this example, select with highest entropy

		// SVM Classification
		int numClassificationRuns = 10;
		SVMClassifier classifier = new SVMClassifier(dataset);
		ClassificationResult result = classifier.evaluate(selectedBands, numClassificationRuns);

		DescriptiveStatistics OA = result.getOverallAccuracy();
		DescriptiveStatistics AOA = result.getAverageOverallAccuracy();

		log.info("OA: " + OA.getMean() + " ( SD:" + OA.getStandardDeviation() + ")");
		log.info("AOA: " + AOA.getMean() + " ( SD:" + AOA.getStandardDeviation() + ")");
	}

	public static void main(String[] args) {
		try {
			new ExampleExperiment().runExperiment();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
