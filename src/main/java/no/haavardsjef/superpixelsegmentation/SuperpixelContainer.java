package no.haavardsjef.superpixelsegmentation;

import boofcv.struct.image.GrayF32;
import boofcv.struct.image.Planar;
import lombok.extern.log4j.Log4j2;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.IntStream;

/**
 * A container for superpixel data.
 * Holds the superpixel map and provides useful methods for working with superpixels.
 */
@Log4j2
public class SuperpixelContainer {

	private INDArray superpixelMap;
	private final INDArray data;
	private INDArray superpixelMeans; // Shape: [numBands, numSuperpixels]
	private int numSuperpixels;

	public SuperpixelContainer(INDArray data, int numSuperpixels, float spatialWeight) {
		this.data = data;
		this.generateSuperpixelMap(numSuperpixels, spatialWeight);
	}


	/**
	 * Converts data into planar image, and then uses boofCV to generate a superpixel map.
	 */
	private void generateSuperpixelMap(int numSuperpixels, float spatialWeight) {
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
		int[] superpixelMap = superpixelSegmentation.segment(image, false, numSuperpixels, spatialWeight);
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
		IntStream.range(0, (int) this.data.shape()[0]).forEach(band -> calculateMeanUsingMap(band));
		// Stop timer
		long endTime = System.currentTimeMillis();
		log.info("Superpixel means calculated in {} ms", endTime - startTime);
	}

	/**
	 * Calculates the mean value for every superpixel in the given band.
	 *
	 * @param bandIndex The index of the band to calculate the mean for.
	 */
	private void calculateMeanUsingMap(int bandIndex) {
		INDArray bandData = this.data.get(NDArrayIndex.point(bandIndex), NDArrayIndex.all(), NDArrayIndex.all());
		int numRows = bandData.rows();
		int numCols = bandData.columns();

		Map<Integer, List<Double>> superpixelPixelValues = new HashMap<>();

		for (int row = 0; row < numRows; row++) {
			for (int col = 0; col < numCols; col++) {
				int superpixelIndex = this.superpixelMap.getInt(row, col);
				double pixelValue = bandData.getDouble(row, col);

				superpixelPixelValues.computeIfAbsent(superpixelIndex, k -> new ArrayList<>()).add(pixelValue);
			}
		}

		for (Map.Entry<Integer, List<Double>> entry : superpixelPixelValues.entrySet()) {
			int superpixelIndex = entry.getKey();
			List<Double> pixelValues = entry.getValue();
			double mean = pixelValues.stream().mapToDouble(Double::doubleValue).average().orElse(0);
			this.superpixelMeans.putScalar(bandIndex, superpixelIndex, mean);
		}
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

	public void saveSPMap(String fileName) {
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
			for (int row = 0; row < this.superpixelMap.rows(); row++) {
				for (int col = 0; col < this.superpixelMap.columns(); col++) {
					writer.write(String.valueOf(this.superpixelMap.getDouble(row, col)));
					if (col < this.superpixelMap.columns() - 1) {
						writer.write(",");
					}
				}
				writer.newLine();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public int getSuperpixelIndex(int row, int col) {
		return this.superpixelMap.getInt(row, col);
	}

	public int getSuperpixelIndex(int pixelIndex) {
		int row = pixelIndex / this.superpixelMap.columns();
		int col = pixelIndex % this.superpixelMap.columns();
		return this.superpixelMap.getInt(row, col);
	}


}
