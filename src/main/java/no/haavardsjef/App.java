package no.haavardsjef;

import no.haavardsjef.experiments.BandSelectionExperiment;
import no.haavardsjef.fcm.FCM;
import no.haavardsjef.fcm.distancemetrics.EuclideanDistance;
import no.haavardsjef.pso.SwarmPopulation;
import no.haavardsjef.utility.DataLoader;
import no.haavardsjef.utility.DataLoaderCSV;
import no.haavardsjef.vizualisation.PlotScatter;
import no.haavardsjef.utility.DataLoader;

import java.util.List;
import java.util.ArrayList;

public class App {


	public static void main(String[] args) {
		BandSelectionExperiment bandSelectionExperiment = new BandSelectionExperiment();
		bandSelectionExperiment.runExperiment();


	}


}
