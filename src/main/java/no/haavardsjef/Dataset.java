package no.haavardsjef;

import lombok.extern.log4j.Log4j2;
import no.haavardsjef.superpixelsegmentation.SuperpixelContainer;
import no.haavardsjef.utility.HyperspectralDataLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.util.List;

/**
 * A container for a dataset. Contains the data, groundtruth, and other metadata.
 * Provides useful methods for working with the dataset.
 */
@Log4j2
public class Dataset {

	public INDArray data;
	public INDArray groundTruth;
	private int numBands;
	private int imageWidth;
	private int imageHeight;
	private int numPixels;
	private int numClasses;
	private String datasetPath;
	private DatasetName datasetName;
	private SuperpixelContainer superpixelContainer;

	public Dataset(String datasetPath, DatasetName datasetName) throws IOException {
		this.datasetPath = datasetPath;
		this.datasetName = datasetName;
		this.load();
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
		log.info("Dataset {} loaded, numBands: {}, imageWidth: {}, imageHeight: {}, numPixels: {}, numClasses: {}", this.datasetName,
				this.numBands, this.imageWidth, this.imageHeight, this.numPixels, this.numClasses);
	}

	/**
	 * Initializes the superpixel container. Must be called before using any superpixel related methods.
	 */
	public void setupSuperpixelContainer() {
		this.superpixelContainer = new SuperpixelContainer(this.data);
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


	public int[] getGroundTruthFlattenedAsArray() {
		return this.groundTruth.ravel().toIntVector();
	}


	public static void main(String[] args) throws IOException {
		Dataset ds = new Dataset("data/indian_pines", DatasetName.indian_pines);
//		ds.setupSuperpixelContainer();
		int[] gt = ds.getGroundTruthFlattenedAsArray();


		double dist = ds.euclideanDistance(0, 1);
		double spDist = ds.euclideanDistanceSP(0, 1);
	}
}
