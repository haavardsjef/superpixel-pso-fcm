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
	 * Calculate the mean value in a given superpixel for a given band index.
	 * This method assumes that the class has already been supplied with the correct HSI-data.
	 * If you have the band data easily available, use the method meanValueInSuperpixelGivenData instead.
	 *
	 * @param superpixelIndex - The index of the superpixel we wish to calculate the mean of.
	 * @param bandIndex       - The index of the band we wish to calculate the mean of across that superpixel.
	 * @return - The mean value for the given superpixel and band.
	 */
	public double meanValueInSuperpixelUsingBandIndex(int superpixelIndex, int bandIndex) {
		if (this.hsiDataFlattened == null) {
			throw new IllegalArgumentException("The class has not been supplied with HSI-data");
		}
		double[] bandData = hsiDataFlattened[bandIndex];
		return meanValueInSuperpixel(superpixelIndex, bandData);
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


	/**
	 * Calculates the mean value in all superpixels from the band data stored in the class.
	 * This method assumes that the class has already been supplied with the correct HSI-data.
	 * If you have the band data easily available, use the method meanValueInAllSuperpixelsGivenData instead.
	 *
	 * @param bandIndex - The index of the band we wish to calculate the mean of across all superpixels.
	 * @return - An array containing the mean value for each superpixel across that band.
	 */
	public double[] meanValueInAllSuperpixelsUsingBandIndex(int bandIndex) {
		if (this.hsiDataFlattened == null) {
			throw new IllegalArgumentException("The class has not been supplied with HSI-data");
		}
		double[] band = hsiDataFlattened[bandIndex];
		return this.meanValueInAllSuperpixels(band);
	}


}
