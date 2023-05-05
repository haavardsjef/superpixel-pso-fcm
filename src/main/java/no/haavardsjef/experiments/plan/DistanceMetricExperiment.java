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
		MLFlow mlFlow = new MLFlow("http://35.185.118.215:8080/");

		// Create a new experiment
		String experimentName = "distance-metric";
		mlFlow.initializeExperiment(experimentName);


		Dataset dataset = new Dataset(DatasetName.Pavia);


		int numSuperpixels = 900;
		float spatialWeight = 10000f;

		dataset.setupSuperpixelContainer(numSuperpixels, spatialWeight);

		//Calculations needed for the different informationt based measures:
		dataset.calculateProbabilityDistributionsSPmean(); //SP_MEAN_KL_Divergence, SP_MEAN_DISJOINT, SP_MEAN_COR_COF
		dataset.calculateDisjointInfoSuperpixelLevel(); //SP_MEAN_DISJOINT
		dataset.calculateCorrelationCoefficients_SP();  //SP_MEAN_COR_COF
		dataset.calculateKlDivergencesSuperpixelLevel(); //SP_LEVEL_KL_DIVERGENCE_L1NORM}


		// Parameters that are the constant
		double fuzziness = 2.0;
		Bounds bounds = dataset.getBounds();

		DistanceMeasure[] distanceMeasures = new DistanceMeasure[]{DistanceMeasure.SP_MEAN_DISJOINT, DistanceMeasure.SP_LEVEL_KL_DIVERGENCE_L1NORM, DistanceMeasure.SP_MEAN_COR_COF};

		// For every distance measure
		for (DistanceMeasure distanceMeasure : distanceMeasures) {

			IObjectiveFunction objectiveFunction = new FuzzyCMeans(dataset, fuzziness, distanceMeasure);
			for (int i = 5; i < 51; i += 5) {


				for (int run = 0; run < 10; run++) {


					int numberOfBandsToSelect = i;
					// Start a new run
					String runName = "PA-" + distanceMeasure + "-" + numberOfBandsToSelect + "-" + run;
					mlFlow.startRun(runName);

					long startTime = System.currentTimeMillis();
					PSOParams params = new PSOParams(numberOfBandsToSelect);


					// PSO-FCM to select cluster centers
					SwarmPopulation swarmPopulation = new SwarmPopulation(params.numParticles, numberOfBandsToSelect, bounds, objectiveFunction);
					Particle solution = swarmPopulation.optimize(params.numIterations, params.w, params.c1, params.c2, false, true);
					float fitness = solution.evaluate();
					mlFlow.logParam("fitness", String.valueOf(fitness));
					List<Integer> clusterCentroids = solution.getDiscretePositionSorted();

					long optimizationEndTime = System.currentTimeMillis();
					long optimizationDuration = (optimizationEndTime - startTime) / 1000;
					mlFlow.logMetric("optimizationDuration", optimizationDuration);

					// Log parameters
					mlFlow.logPSOParams(params);
					mlFlow.logParam("numSuperpixels", String.valueOf(numSuperpixels));
					mlFlow.logParam("spatialWeight", String.valueOf(spatialWeight));
					mlFlow.logParam("fuzziness", String.valueOf(fuzziness));
					mlFlow.logParam("dataset", dataset.getDatasetName().toString());
					mlFlow.logParam("distanceMeasure", distanceMeasure.toString());
					mlFlow.logParam("clusterCentroids", clusterCentroids.toString());
					mlFlow.logParam("numIterationsRan", String.valueOf(swarmPopulation.numIterationsRan));


					// End run
					mlFlow.endRun();
				}
			}
		}
	}


//	}

	public static void main(String[] args) throws IOException {
		DistanceMetricExperiment distanceMetricExperiment = new DistanceMetricExperiment();
		distanceMetricExperiment.runExperiment();
	}
}
