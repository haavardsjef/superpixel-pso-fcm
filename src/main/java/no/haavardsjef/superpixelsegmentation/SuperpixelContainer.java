package no.haavardsjef.superpixelsegmentation;

import boofcv.struct.image.GrayF32;
import boofcv.struct.image.Planar;
import lombok.extern.log4j.Log4j2;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

@Log4j2
public class SuperpixelContainer {

	private INDArray superpixelMap;
	private INDArray data;
	private INDArray superpixelMeans; // Shape: [numSuperpixels, numBands]
	private int numSuperpixels;

	public SuperpixelContainer(INDArray data) {
		this.data = data;
	}


	public void generateSuperpixelMap() {
		int numBands = (int) this.data.shape()[0];
		int imageWidth = (int) this.data.shape()[2];
		int imageHeight = (int) this.data.shape()[1];
		int numPixels = imageWidth * imageHeight;
		// Convert data to Planar Image
		Planar<GrayF32> image = new Planar<>(GrayF32.class, imageWidth, imageHeight, numBands);
		for (int i = 0; i < numPixels; i++) {
			// Get row and col
			int row = i / imageWidth;
			int col = i % imageWidth;

			for (int j = 0; j < numBands; j++) {
				image.getBand(j).set(col, row, (float) this.data.getDouble(j, row, col));
			}
		}
		log.info("Planar image created");
		SuperpixelSegmentation superpixelSegmentation = new SuperpixelSegmentation();
		int[] superpixelMap = superpixelSegmentation.segment(image, true);
		log.info("Superpixel map created");
		this.superpixelMap = Nd4j.createFromArray(superpixelMap).reshape(imageHeight, imageWidth);
		this.numSuperpixels = Arrays.stream(superpixelMap).max().getAsInt() + 1;
		superpixelMeans = Nd4j.zeros(this.numSuperpixels, numBands);
	}


	public void calculateSuperpixelMeans() {
		// TODO: Precalculate the mean for every band in every superpixel
		log.info("Calculating superpixel means");
	}

}
