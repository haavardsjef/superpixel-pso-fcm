package no.haavardsjef.experiments;

import boofcv.struct.image.GrayF32;
import boofcv.struct.image.Planar;
import no.haavardsjef.superpixelsegmentation.PCA_Implementation;
import no.haavardsjef.superpixelsegmentation.SuperpixelSegmentation;
import no.haavardsjef.utility.DataLoader;
import no.haavardsjef.utility.FileWriter;
import org.nd4j.linalg.api.ndarray.INDArray;

public class SuperpixelSegmentationExperiment implements IExperiment {

	@Override
	public void runExperiment() {

		DataLoader dl = new DataLoader();
		dl.loadData();
		double[][] hsiDataFlattened = dl.getDataFlatted();

		INDArray principleComponents = PCA_Implementation.performPCA(hsiDataFlattened, true);
		System.out.println(principleComponents);

		// For each principle component
		for (int i = 0; i < principleComponents.columns(); i++) {
			// Get min value
			double min = principleComponents.getColumns(i).minNumber().doubleValue();
			// Subtract min for each value to shift to positive
			principleComponents.putColumn(i, principleComponents.getColumns(i).sub(min));
			// Get max value
			double max = principleComponents.getColumns(i).maxNumber().doubleValue();
			// Multiply each value by 255/max to scale to 0-255
			principleComponents.putColumn(i, principleComponents.getColumns(i).mul(255 / max));
		}


		// Create Planar image from principle components
		Planar<GrayF32> image = new Planar<GrayF32>(GrayF32.class, 145, 145, 3); // TODO: Automatic width and height


		for (int i = 0; i < principleComponents.rows(); i++) {
			// Get row and col
			int row = i / 145;
			int col = i % 145;

			image.getBand(0).set(col, row, (float) principleComponents.getDouble(i, 0));
			image.getBand(1).set(col, row, (float) principleComponents.getDouble(i, 1));
			image.getBand(2).set(col, row, (float) principleComponents.getDouble(i, 2));
		}
		System.out.println("Planar image created");


		SuperpixelSegmentation superpixelSegmentation = new SuperpixelSegmentation();
		int[] superpixelMap = superpixelSegmentation.segment(image, false);

		// Save superpixel map to file
		FileWriter.csvWrite(superpixelMap, "segmentation/map.csv");

	}

	public static void main(String[] args) {
		SuperpixelSegmentationExperiment superpixelSegmentationExperiment = new SuperpixelSegmentationExperiment();
		superpixelSegmentationExperiment.runExperiment();
	}
}
