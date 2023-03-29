package no.haavardsjef.superpixelsegmentation;

import boofcv.struct.image.GrayF32;
import boofcv.struct.image.Planar;
import lombok.extern.log4j.Log4j2;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * A container for superpixel data.
 * Holds the superpixel map and provides useful methods for working with superpixels.
 */
@Log4j2
public class SuperpixelContainer {

	private INDArray superpixelMap;
	private INDArray data;
	private INDArray superpixelMeans; // Shape: [numBands, numSuperpixels]
	private int numSuperpixels;

	public SuperpixelContainer(INDArray data) {
		this.data = data;
		this.generateSuperpixelMap();
	}


	/**
	 * Converts data into planar image, and then uses boofCV to generate a superpixel map.
	 */
	private void generateSuperpixelMap() {
		log.info("Generating superpixel map");
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
		int[] superpixelMap = superpixelSegmentation.segment(image, false);
		this.superpixelMap = Nd4j.createFromArray(superpixelMap).reshape(imageHeight, imageWidth);
		this.numSuperpixels = Arrays.stream(superpixelMap).max().getAsInt() + 1;
		log.info("Superpixel map created");
		this.calculateSuperpixelMeans();
	}


	/**
	 * Uses to superpixel map to calculate the mean value of each superpixel for each band,
	 * and stores the result in superpixelMeans.
	 */
	private void calculateSuperpixelMeans() {
		log.info("Calculating superpixel means");
		// Start timer
		long startTime = System.currentTimeMillis();
		this.superpixelMeans = Nd4j.zeros((int) this.data.shape()[0], this.numSuperpixels);
		IntStream.range(0, (int) this.data.shape()[0]).parallel().forEach(band -> calculateMean(band));
		// Stop timer
		long endTime = System.currentTimeMillis();
		log.info("Superpixel means calculated in {} ms", endTime - startTime);

	}

	/**
	 * Calculates the mean value for every superpixel in the given band.
	 *
	 * @param bandIndex The index of the band to calculate the mean for.
	 */
	private void calculateMean(int bandIndex) {
		INDArray bandData = this.data.get(NDArrayIndex.point(bandIndex), NDArrayIndex.all(), NDArrayIndex.all());
		IntStream.range(0, this.numSuperpixels).forEach(superpixelIndex -> {
			INDArray result = bandData.mul(this.superpixelMap.eq(superpixelIndex));

			// Create a binary mask for non-zero elements
			INDArray nonZeroMask = result.dup();
			BooleanIndexing.replaceWhere(nonZeroMask, 1, Conditions.notEquals(0));
			BooleanIndexing.replaceWhere(nonZeroMask, 0, Conditions.equals(0));

			double mean = result.sumNumber().doubleValue() / nonZeroMask.sumNumber().intValue();
			// TODO: Also calculate the median.
			this.superpixelMeans.putScalar(bandIndex, superpixelIndex, mean);

		});
	}

	/**
	 * Get the mean value for each superpixel in a given band.
	 *
	 * @param bandIndex The index of the band to get the superpixel means for.
	 * @return An INDArray of shape [numSuperpixels] containing the mean value for each superpixel.
	 */
	public INDArray getSuperpixelMeans(int bandIndex) {
		return this.superpixelMeans.get(NDArrayIndex.point(bandIndex), NDArrayIndex.all());
	}

	public int getNumSuperpixels() {
		return numSuperpixels;
	}
}
