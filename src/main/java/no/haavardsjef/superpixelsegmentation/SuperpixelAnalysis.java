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
