package no.haavardsjef.experiments;

import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;
import no.haavardsjef.classification.SVMClassifier;
import no.haavardsjef.fcm.FuzzyCMeans;
import no.haavardsjef.objectivefunctions.IObjectiveFunction;
import no.haavardsjef.pso.Particle;
import no.haavardsjef.pso.SwarmPopulation;
import no.haavardsjef.utility.Bounds;
import no.haavardsjef.vizualisation.PlotLine;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.IntStream;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.mlflow.tracking.*;
import org.mlflow.api.proto.Service.*;

public class BandSelectionExperiment implements IExperiment {


	public void runExperiment() throws IOException {
//		IObjectiveFunction objectiveFunction = new FCM(2.0f, new EuclideanDistance(), new DataLoader());
		Dataset dataset = new Dataset(DatasetName.indian_pines);
		IObjectiveFunction objectiveFunction = new FuzzyCMeans(dataset, 2.0);
		Bounds bounds = dataset.getBounds();


		// Initialize new MLflow client to connect to local MLflow server
		MLFlow mlFlow = new MLFlow();

		// Create a new experiment
		String experimentName = "band-selection-classification";
		mlFlow.initializeExperiment(experimentName);


		for (int i = 1; i < 30; i++) {

			int numberOfBandsToSelect = i;
			int numParticles = 100;
			int numIterations = 50;
			float w = 0.5f;
			float c1 = 0.5f;
			float c2 = 0.2f;
			int numClassificationRuns = 10;

			// Start a new run

			String runName = numberOfBandsToSelect + "-bands";
			mlFlow.startRun(runName);

			// Actual run
			mlFlow.logParam("numBands", String.valueOf(numberOfBandsToSelect));

			SwarmPopulation swarmPopulation = new SwarmPopulation(numParticles, numberOfBandsToSelect, bounds, objectiveFunction);
			Particle solution = swarmPopulation.optimize(numIterations, w, c1, c2, false);

			List<Integer> selectedBands = solution.getDiscretePositionSorted();

			// Log parameters
			mlFlow.logParam("selectedBands", selectedBands.toString());
			mlFlow.logParam("numParticles", String.valueOf(numParticles));
			mlFlow.logParam("numIterations", String.valueOf(numIterations));
			mlFlow.logParam("w", String.valueOf(w));
			mlFlow.logParam("c1", String.valueOf(c1));
			mlFlow.logParam("c2", String.valueOf(c2));
			mlFlow.logParam("numClassificationRuns", String.valueOf(numClassificationRuns));

			// Log metrics

			System.out.println("Selected bands:" + selectedBands);

			SVMClassifier svmClassifier = new SVMClassifier(dataset);

			DescriptiveStatistics stats = svmClassifier.evaluate(selectedBands, numClassificationRuns);
			mlFlow.logMetric("accuracy", stats.getMean());
			mlFlow.logMetric("std", stats.getStandardDeviation());

			mlFlow.endRun();

		}
	}


	public static void main(String[] args) throws IOException {
		BandSelectionExperiment bandSelectionExperiment = new BandSelectionExperiment();
		bandSelectionExperiment.runExperiment();
	}


}
