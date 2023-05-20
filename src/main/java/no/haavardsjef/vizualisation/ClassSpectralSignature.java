package no.haavardsjef.vizualisation;

import no.haavardsjef.dataset.Dataset;
import no.haavardsjef.dataset.DatasetName;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ClassSpectralSignature {

	public static void main(String[] args) throws IOException {
		Dataset ds = new Dataset(DatasetName.Pavia);

		// Get ground truth
		int[] gt = ds.getGroundTruthFlattenedAsArray();

		int classToVisualize = 2;

		// Get pixelIndex of all samples of class `classToVisualize`
		List<Integer> pixelIndexes = new ArrayList<>();


		for (int i = 0; i < gt.length; i++) {
			if (gt[i] == classToVisualize) {
				pixelIndexes.add(i);
			}
		}

		System.out.println(pixelIndexes);

		List<List<Double>> spectralSignatures = new ArrayList<>();

		// Get spectral intesity of all samples of class `classToVisualize`
		pixelIndexes.stream().forEach(pixelIndex -> {
			INDArray d = ds.getPixelIntensity(pixelIndex);
			System.out.println(d);
			List<Double> spectralSignature = new ArrayList<>();
			for (int i = 0; i < d.length(); i++) {
				spectralSignature.add(d.getDouble(i));
			}
			spectralSignatures.add(spectralSignature);

		});

		PlotLine.plotMultiple(spectralSignatures);
	}
}
