package no.haavardsjef.experiments.plan;

import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;
import no.haavardsjef.experiments.IExperiment;
import no.haavardsjef.vizualisation.PlotLine;

import java.io.IOException;
import java.util.ArrayList;

public class DistanceMetricExperiment implements IExperiment {

	public void runExperiment() throws IOException {
		Dataset dataset = new Dataset(DatasetName.indian_pines);
		dataset.setupSuperpixelContainer();

		ArrayList<Double> euclideanDistance = new ArrayList<>();
		ArrayList<Double> euclideanDistanceSP = new ArrayList<>();

		int numPixels = dataset.getNumPixels();
		int numSuperpixels = dataset.getNumSuperpixels();

		for (int i = 0; i < 199; i++) {
			euclideanDistance.add(dataset.euclideanDistance(i, i + 1));
			euclideanDistanceSP.add(dataset.euclideanDistanceSP(i, i + 1));
		}

		PlotLine.plotMultiple(euclideanDistance, euclideanDistanceSP);
		PlotLine.saveToCsv(euclideanDistance, euclideanDistanceSP);


		PlotLine.barChart(euclideanDistance);


	}

	public static void main(String[] args) throws IOException {
		DistanceMetricExperiment distanceMetricExperiment = new DistanceMetricExperiment();
		distanceMetricExperiment.runExperiment();
	}
}
