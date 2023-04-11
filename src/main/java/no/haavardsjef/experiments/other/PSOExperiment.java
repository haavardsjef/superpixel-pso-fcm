package no.haavardsjef.experiments.other;

import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;
import no.haavardsjef.experiments.IExperiment;
import no.haavardsjef.fcm.FuzzyCMeans;
import no.haavardsjef.objectivefunctions.IObjectiveFunction;
import no.haavardsjef.pso.Particle;
import no.haavardsjef.pso.SwarmPopulation;
import no.haavardsjef.utility.Bounds;
import no.haavardsjef.utility.DistanceMeasure;

import java.io.IOException;

public class PSOExperiment implements IExperiment {

	public void runExperiment() throws IOException {
		Dataset dataset = new Dataset(DatasetName.indian_pines);
		IObjectiveFunction objectiveFunction = new FuzzyCMeans(dataset, 2.0, DistanceMeasure.PIXEL_EUCLIDEAN);
		Bounds bounds = dataset.getBounds();
		SwarmPopulation swarmPopulation = new SwarmPopulation(100, 2, bounds, objectiveFunction);
		Particle solution = swarmPopulation.optimize(50, 0.8f, 1.0f, 1.0f, true, false);
	}


	public static void main(String[] args) throws IOException {
		PSOExperiment psoExperiment = new PSOExperiment();
		psoExperiment.runExperiment();
	}
}
