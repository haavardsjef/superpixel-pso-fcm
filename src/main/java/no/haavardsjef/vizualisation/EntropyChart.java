package no.haavardsjef.vizualisation;

import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;

import java.io.IOException;
import java.util.List;

public class EntropyChart {

	public static void main(String[] args) throws IOException {
		Dataset ds_ip = new Dataset(DatasetName.indian_pines, true);
		Dataset ds_sa = new Dataset(DatasetName.Salinas, true);
		Dataset ds_pa = new Dataset(DatasetName.Pavia, true);


		List<Double> entropies_ip = ds_ip.getEntropies();
		List<Double> entropies_sa = ds_sa.getEntropies();
		List<Double> entropies_pa = ds_pa.getEntropies();

		PlotLine.saveToCsv(entropies_ip, entropies_sa, entropies_pa); // Save to csv for plotting in python


	}
}
