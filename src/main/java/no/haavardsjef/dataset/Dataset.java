package no.haavardsjef.dataset;

import com.google.common.math.DoubleMath;
import lombok.extern.log4j.Log4j2;
import no.haavardsjef.superpixelsegmentation.SuperpixelContainer;
import no.haavardsjef.utility.Bounds;
import no.haavardsjef.utility.DistanceMeasure;
import no.haavardsjef.utility.HyperspectralDataLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A container for a dataset. Contains the data, groundtruth, and other metadata.
 * Provides useful methods for working with the dataset.
 */
@Log4j2
public class Dataset implements IDataset {

	private INDArray data;
	public INDArray groundTruth;
	private int numBands;
	private int imageWidth;
	private int imageHeight;
	private int numPixels;
	private int numClasses;
	private final String datasetPath;
	private final DatasetName datasetName;
	private SuperpixelContainer superpixelContainer;
	private Bounds bounds;
	private List<Double> entropies;

	private double[][] probabilityDistributions;

	public Dataset(DatasetName datasetName) throws IOException {
		this.datasetPath = "data/" + datasetName;
		this.datasetName = datasetName;
		this.load();
		this.calculateProbabilityDistributions();
		this.calculateEntropies();
	}


	/**
	 * Loads the dataset from the given path.
	 *
	 * @throws IOException if the dataset cannot be loaded
	 */
	private void load() throws IOException {
		String correctedDataPath = this.datasetPath + "/" + this.datasetName + "_corrected.mat";
		String groundTruthPath = this.datasetPath + "/" + this.datasetName + "_gt.mat";
		this.data = HyperspectralDataLoader.loadData(correctedDataPath);
		this.groundTruth = HyperspectralDataLoader.loadGroundTruth(groundTruthPath);
		this.numBands = (int) this.data.shape()[0];
		this.imageWidth = (int) this.data.shape()[2];
		this.imageHeight = (int) this.data.shape()[1];
		this.numPixels = this.imageWidth * this.imageHeight;
		this.numClasses = this.groundTruth.maxNumber().intValue() + 1;
		this.bounds = new Bounds(0, this.numBands - 1);
		log.info("Dataset {} loaded, numBands: {}, imageWidth: {}, imageHeight: {}, numPixels: {}, numClasses: {}", this.datasetName,
				this.numBands, this.imageWidth, this.imageHeight, this.numPixels, this.numClasses);

	}

	/**
	 * Initializes the superpixel container. Must be called before using any superpixel related methods.
	 */
	public void setupSuperpixelContainer(int numSuperpixels, float spatialWeight) {
		this.superpixelContainer = new SuperpixelContainer(this.data, numSuperpixels, spatialWeight);
	}

	public void setupSuperpixelContainer() {
		this.superpixelContainer = new SuperpixelContainer(this.data, 100, 200f);
	}

	/**
	 * Calculates the euclidean distance between two bands, using all pixels.
	 *
	 * @param bandIndex1 The index of the first band.
	 * @param bandIndex2 The index of the second band.
	 * @return The euclidean distance between the two bands.
	 */
	public double euclideanDistance(int bandIndex1, int bandIndex2) {
		INDArray bandData1 = this.data.get(NDArrayIndex.point(bandIndex1), NDArrayIndex.all(), NDArrayIndex.all());
		INDArray bandData2 = this.data.get(NDArrayIndex.point(bandIndex2), NDArrayIndex.all(), NDArrayIndex.all());

		return bandData1.distance2(bandData2); // Returns the euclidean distance.
	}

	@Override
	public INDArray getData() {
		return this.data;
	}

	@Override
	public Bounds getBounds() {
		return this.bounds;
	}

	public int getNumSuperpixels() {
		return superpixelContainer.getNumSuperpixels();
	}

	public int getNumPixels() {
		return this.numPixels;
	}

	public DatasetName getDatasetName() {
		return datasetName;
	}

	/**
	 * Calculates the euclidean distance between two bands, using mean of superpixels.
	 *
	 * @param bandIndex1 The index of the first band.
	 * @param bandIndex2 The index of the second band.
	 * @return The euclidean distance between the two bands.
	 */
	public double euclideanDistanceSP(int bandIndex1, int bandIndex2) {
		if (this.superpixelContainer == null) {
			throw new IllegalStateException("SuperpixelContainer is not initialized.");
		}
		INDArray bandData1 = this.superpixelContainer.getSuperpixelMeans(bandIndex1);
		INDArray bandData2 = this.superpixelContainer.getSuperpixelMeans(bandIndex2);

		return bandData1.distance2(bandData2); // Returns the euclidean distance.
	}

	public double[][] getDataAsArray() {
		int firstDim = (int) this.data.size(0);
		int flattenedSize = (int) (this.data.size(1) * this.data.size(2));

		double[][] result = new double[firstDim][];

		// Iterate through the first dimension
		for (int i = 0; i < firstDim; i++) {
			// Get the 2D tensor along the first dimension
			INDArray tensor = this.data.tensorAlongDimension(i, 1, 2);

			// Flatten the 2D tensor and store it in the result array
			result[i] = tensor.reshape(flattenedSize).toDoubleVector();
		}
		return result;
	}

	public INDArray getBand(int bandIndex) {
		return this.data.get(NDArrayIndex.point(bandIndex), NDArrayIndex.all(), NDArrayIndex.all());
	}

	public INDArray getBandFlattened(int bandIndex) {
		return this.data.get(NDArrayIndex.point(bandIndex), NDArrayIndex.all(), NDArrayIndex.all()).reshape(this.numPixels);
	}

	public INDArray getBands(List<Integer> bandIndexes) {
		long[] indices = bandIndexes.stream().mapToLong(i -> i).toArray();
		return this.data.get(NDArrayIndex.indices(indices), NDArrayIndex.all(), NDArrayIndex.all());
	}

	public INDArray getBandsFlattened(List<Integer> bandIndexes) {
		INDArray dataFlattened = this.data.reshape(this.numBands, this.numPixels);
		return dataFlattened.get(NDArrayIndex.indices(bandIndexes.stream().mapToLong(i -> i).toArray()), NDArrayIndex.all());
	}

	public INDArray getBandMax() {
		return this.data.max(1, 2);
	}

	public INDArray getBandMin() {
		return this.data.min(1, 2);
	}


	public int[] getGroundTruthFlattenedAsArray() {
		return this.groundTruth.ravel().toIntVector();
	}

	public int getNumBands() {
		return this.numBands;
	}

	public void calculateEntropies() {
		log.info("Calculating entropies for dataset {}...", this.datasetName);
		List<Double> entropies = new ArrayList<>();

		for (int bandIndex = 0; bandIndex < this.numBands; bandIndex++) {
			entropies.add(this.calculateEntropy(bandIndex));
		}

		this.entropies = entropies;
	}


	public void calculateProbabilityDistributions() {
		log.info("Calculating probability distributions for dataset {}...", this.datasetName);
		this.probabilityDistributions = new double[this.numBands][256];
		for (int bandIndex = 0; bandIndex < this.numBands; bandIndex++) {
			int NUM_BINS = 256;
			double[] r = this.getBandFlattened(bandIndex).toDoubleVector();

			int[] histogram = new int[NUM_BINS];
			double[] normalHistogram = new double[NUM_BINS];

			double min = (double) this.getBand(bandIndex).minNumber();
			double max = (double) this.getBand(bandIndex).maxNumber();

			// Bin all pixels to create histogram
			for (double p : r) {
				int bin = (int) Math.floor((p - min) / (max - min) * (NUM_BINS - 1));
				histogram[bin] += 1;
			}

			// Normalize histogram into probability distribution
			for (int i = 0; i < NUM_BINS; i++) {
				normalHistogram[i] = (double) histogram[i] / (double) (this.imageWidth * this.imageHeight);
			}

			this.probabilityDistributions[bandIndex] = normalHistogram;
		}
	}

	private double calculateEntropy(int band) {
		return -Arrays.stream(probabilityDistributions[band]).reduce(0.0, (acc, val) -> {
			if (val == 0.0) {
				return acc;
			} else {
				return acc + (val * DoubleMath.log2(val));
			}
		});
	}

	public List<Double> getEntropies() {
		return this.entropies;
	}

	public double distance(DistanceMeasure distanceMeasure, int bandIndex1, int bandIndex2) {
		switch (distanceMeasure) {
			case PIXEL_EUCLIDEAN:
				return this.euclideanDistance(bandIndex1, bandIndex2);
			case SP_MEAN_EUCLIDEAN:
				return this.euclideanDistanceSP(bandIndex1, bandIndex2);
			default:
				throw new IllegalArgumentException("Unknown distance measure: " + distanceMeasure);
		}
	}

	public void saveSuperpixelMap(String path) throws IOException {
		if (this.superpixelContainer == null) {
			throw new IllegalStateException("SuperpixelContainer is not initialized.");
		}
		this.superpixelContainer.saveSPMap(path);
	}

	public int getSuperpixelIndex(int pixelIndex) {
		if (this.superpixelContainer == null) {
			throw new IllegalStateException("SuperpixelContainer is not initialized.");
		}
		return this.superpixelContainer.getSuperpixelIndex(pixelIndex);
	}


	public static void main(String[] args) throws IOException {
		Dataset ds = new Dataset(DatasetName.indian_pines);
		ds.calculateProbabilityDistributions();
	}
}
