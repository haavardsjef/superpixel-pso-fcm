package no.haavardsjef.superpixelsegmentation;


import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Helper class to perform analysis on superpixel segmentation.
 */
public class SuperpixelAnalysis {
	private int[] superpixelMap;
	private double[][] hsiDataFlattened;


	private int numberOfSuperpixels;

	public SuperpixelAnalysis(int[] superpixelMap, double[][] hsiDataFlattened) {
		this.superpixelMap = superpixelMap;
		this.hsiDataFlattened = hsiDataFlattened;

		// Get number of superpixels
		numberOfSuperpixels = Arrays.stream(superpixelMap).max().getAsInt() + 1;

	}

	public SuperpixelAnalysis(int[] superpixelMap) {
		this.superpixelMap = superpixelMap;
		// Get number of superpixels
		numberOfSuperpixels = Arrays.stream(superpixelMap).max().getAsInt() + 1;

	}

	/**
	 * @return - The number of superpixels in the superpixel map.
	 */
	public int getNumberOfSuperpixels() {
		return numberOfSuperpixels;
	}

	/**
	 * Calculate the mean value in a given superpixel for the given band data.
	 *
	 * @param superpixelIndex - The index of the superpixel we wish to calculate the mean of.
	 * @param bandData        - The flattened pixel values from a single band.
	 * @return - The mean value for the given superpixel and band.
	 */
	public double meanValueInSuperpixel(int superpixelIndex, double[] bandData) {
		if (superpixelMap.length != bandData.length) {
			throw new IllegalArgumentException("The superpixel map and the band data must have the same length.");
		}
		// Sum all pixel values for the specific band if in correct superpixel
		int[] pixelIndexes = IntStream.range(0, bandData.length).filter(i -> superpixelMap[i] == superpixelIndex).toArray();
		double totalValue = Arrays.stream(pixelIndexes).mapToObj(i -> bandData[i]).reduce(0.0, (a, b) -> a + b).doubleValue();

		return totalValue / pixelIndexes.length;
	}

	/**
	 * Calculates the mean value in all superpixels from the given band data.
	 *
	 * @param bandData - The flattened pixel values from a single band.
	 * @return - An array containing the mean value for each superpixel across that band.
	 */
	public double[] meanValueInAllSuperpixels(double[] bandData) {
		if (superpixelMap.length != bandData.length) {
			throw new IllegalArgumentException("The superpixel map and the band data must have the same length.");
		}
		double[] meanValues = new double[numberOfSuperpixels];
		for (int i = 0; i < numberOfSuperpixels; i++) {
			meanValues[i] = meanValueInSuperpixel(i, bandData);
		}
		return meanValues;
	}
