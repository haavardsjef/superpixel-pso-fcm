package no.haavardsjef;

import no.haavardsjef.experiments.BandSelectionExperiment;

import java.io.IOException;

public class App {

	public static final String DIR = "C:/Users/haavahje/github/superpixel-pso-fcm";


	public static void main(String[] args) throws IOException {
		BandSelectionExperiment bandSelectionExperiment = new BandSelectionExperiment();
		bandSelectionExperiment.runExperiment();


	}


}
