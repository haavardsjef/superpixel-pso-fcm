package no.haavardsjef.dataset;

import com.google.common.math.DoubleMath;
import lombok.extern.log4j.Log4j2;
import no.haavardsjef.superpixelsegmentation.SuperpixelContainer;
import no.haavardsjef.utility.Bounds;
import no.haavardsjef.utility.DistanceMeasure;
import no.haavardsjef.utility.HyperspectralDataLoader;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

/**
 * A container for a dataset. Contains the data, groundtruth, and other metadata.
 * Provides useful methods for working with the dataset.
 */
@Log4j2
public class Dataset implements IDataset {

	private INDArray data;
	private double[][][] dataAsArray;
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

	private double[][] probabilityDistributionsSPmean;
	private double[][] correlationCoefficientsSPmean;
	private double[][][] probabilityDistributionsSP;
	private double[][] KlDivergencesSuperpixelLevel;
	private double[][] DisjointInfosSuperpixelLevel;
	private final double[][] pixelEuclideanDistances;

	public Dataset(DatasetName datasetName) throws IOException {
		this.datasetPath = "data/" + datasetName;
		this.datasetName = datasetName;
		this.load(true);
		this.calculateProbabilityDistributions();
		this.calculateEntropies();

		pixelEuclideanDistances = new double[numBands][numBands];
	}

	public Dataset(DatasetName datasetName, boolean corrected) throws IOException {
		this.datasetPath = "data/" + datasetName;
		this.datasetName = datasetName;
		this.load(corrected);
		this.calculateProbabilityDistributions();
		this.calculateEntropies();
		pixelEuclideanDistances = new double[numBands][numBands];

	}


	/**
	 * Loads the dataset from the given path.
	 *
	 * @throws IOException if the dataset cannot be loaded
	 */
	private void load(boolean corrected) throws IOException {
		String correctedDataPath = "";
		if (corrected) {
			correctedDataPath = this.datasetPath + "/" + this.datasetName + "_corrected.mat";
		} else {
			correctedDataPath = this.datasetPath + "/" + this.datasetName + ".mat";
		}
		String groundTruthPath = this.datasetPath + "/" + this.datasetName + "_gt.mat";
		this.data = HyperspectralDataLoader.loadData(correctedDataPath);
		// Also convert to 3D array for easier access.
		this.dataAsArray = new double[(int) this.data.shape()[0]][(int) this.data.shape()[1]][(int) this.data.shape()[2]];
		for (int i = 0; i < this.data.shape()[0]; i++) {
			for (int j = 0; j < this.data.shape()[1]; j++) {
				for (int k = 0; k < this.data.shape()[2]; k++) {
					this.dataAsArray[i][j][k] = this.data.getDouble(i, j, k);
				}
			}
		}
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
//	public double euclideanDistance(int bandIndex1, int bandIndex2) {
//		INDArray bandData1 = this.data.get(NDArrayIndex.point(bandIndex1), NDArrayIndex.all(), NDArrayIndex.all());
//		INDArray bandData2 = this.data.get(NDArrayIndex.point(bandIndex2), NDArrayIndex.all(), NDArrayIndex.all());
//
//		return bandData1.distance2(bandData2); // Returns the euclidean distance.
//	}
	public double euclideanDistance(int bandIndex1, int bandIndex2) {
		if (pixelEuclideanDistances[bandIndex1][bandIndex2] == 0.0) {
			double[][] bandData1 = this.dataAsArray[bandIndex1];
			double[][] bandData2 = this.dataAsArray[bandIndex2];

			pixelEuclideanDistances[bandIndex1][bandIndex2] = calculateEuclideanDistance(bandData1, bandData2);
		}
		return pixelEuclideanDistances[bandIndex1][bandIndex2];
	}

//	public void precomputeEuclideanDistances() {
//		for (int i = 0; i < numBands; i++) {
//			for (int j = 0; j < numBands; j++) {
//				if (i == j) {
//					pixelEuclideanDistances[i][j] = 0.0;
//					continue;
//				}
//
//
//				double[][] bandData1 = this.dataAsArray[i];
//				double[][] bandData2 = this.dataAsArray[j];
//
//				pixelEuclideanDistances[i][j] = calculateEuclideanDistance(bandData1, bandData2);
//			}
//		}
//	}

	private double calculateEuclideanDistance(double[][] bandData1, double[][] bandData2) {
		int numRows = bandData1.length;
		int numCols = bandData1[0].length;

		double sum = 0.0;
		for (int row = 0; row < numRows; row++) {
			for (int col = 0; col < numCols; col++) {
				double diff = bandData1[row][col] - bandData2[row][col];
				sum += diff * diff;
			}
		}
		return Math.sqrt(sum);
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

	public INDArray getSuperPixelMap() {
		return superpixelContainer.getSuperixelmap();
	}

	/**
	 * Calculates the euclidean distance between two bands, using mean of superpixels.
	 *
	 * @param bandIndex1 The index of the first band.
	 * @param bandIndex2 The index of the second band.
	 * @return The euclidean distance between the two bands.
	 */
//	public double euclideanDistanceSP(int bandIndex1, int bandIndex2) {
//		if (this.superpixelContainer == null) {
//			throw new IllegalStateException("SuperpixelContainer is not initialized.");
//		}
//		INDArray bandData1 = this.superpixelContainer.getSuperpixelMeans(bandIndex1);
//		INDArray bandData2 = this.superpixelContainer.getSuperpixelMeans(bandIndex2);
//
//		return bandData1.distance2(bandData2); // Returns the euclidean distance.
//	}
	public double euclideanDistanceSP(int bandIndex1, int bandIndex2) {
		if (this.superpixelContainer == null) {
			throw new IllegalStateException("SuperpixelContainer is not initialized.");
		}

		double[] bandData1 = this.superpixelContainer.getSuperpixelMeansArr(bandIndex1);
		double[] bandData2 = this.superpixelContainer.getSuperpixelMeansArr(bandIndex2);

		return euclideanDistance(bandData1, bandData2);
	}

	private double euclideanDistance(double[] vector1, double[] vector2) {
		if (vector1.length != vector2.length) {
			throw new IllegalArgumentException("Input vectors must have the same length.");
		}

		double sum = 0.0;
		for (int i = 0; i < vector1.length; i++) {
			double diff = vector1[i] - vector2[i];
			sum += diff * diff;
		}
		return Math.sqrt(sum);
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

	/**
	 * Calculates the KL-divergence distance between two bands, using all pixels.
	 *
	 * @param bandIndex1 The index of the first band.
	 * @param bandIndex2 The index of the second band.
	 * @return The KL-divergence distance between the two bands.
	 */
	public double KlDivergenceDistance(int bandIndex1, int bandIndex2) {
		return this.calculateKlDivergence(bandIndex1, bandIndex2) + calculateKlDivergence(bandIndex2, bandIndex1);
	}

	/**
	 * Calculates the KL-divergence distance between two bands, using superpixel means.
	 *
	 * @param bandIndex1 The index of the first band.
	 * @param bandIndex2 The index of the second band.
	 * @return The KL-divergence distance between the two bands.
	 */
	public double KlDivergenceDistanceSP(int bandIndex1, int bandIndex2) {
		return this.calculateKlDivergenceSPmean(bandIndex1, bandIndex2) + calculateKlDivergenceSPmean(bandIndex2, bandIndex1);

	}

	/**
	 * Calculates the SuperpixelLevel KlDivergence-L1norm Distances distance between two bands, using mean of superpixels.
	 *
	 * @param bandIndex1 The index of the first band.
	 * @param bandIndex2 The index of the second band.
	 * @return SuperpixelLevel KlDivergence-L1norm distance between two bands.
	 */
	private double KlDivergencesSuperpixelLevelL1normDistance(int bandIndex1, int bandIndex2) {
		if (this.KlDivergencesSuperpixelLevel.length == 0) {
			throw new IllegalStateException("ProbabilityDistributions for KlDivergencesSuperpixelLevel is not calculated.");
		}
		return this.KlDivergencesSuperpixelLevel[bandIndex1][bandIndex2];
	}

	private double calculateKlDivergence(int bandIndex1, int bandIndex2) {

		double[] probDistBand1 = this.probabilityDistributions[bandIndex1];
		double[] probDistBand2 = this.probabilityDistributions[bandIndex2];

		int NUM_BINS = 256;

		double kl = IntStream.range(0, NUM_BINS).mapToDouble(i -> {
			if (probDistBand1[i] == 0.0 || probDistBand2[i] == 0.0) {
				return 0;
			}
			return probDistBand1[i] * DoubleMath.log2(probDistBand1[i] / probDistBand2[i]);
		}).sum();
		return kl;
	}


	/**
	 * Calculates the SuperpixelLevel KlDivergence-L1norm distances for all bands.
	 */
	public void calculateKlDivergencesSuperpixelLevel() {

		this.calculateProbabilityDistributionsSP();

		this.KlDivergencesSuperpixelLevel = new double[this.numBands][this.numBands];
		for (int bandindex1 = 0; bandindex1 < this.numBands; bandindex1++) {
			for (int bandindex2 = 0; bandindex2 < this.numBands; bandindex2++) {
				this.KlDivergencesSuperpixelLevel[bandindex1][bandindex2] = calculateKlDivergenceSP(bandindex1, bandindex2);
			}
		}
	}

	private double calculateKlDivergenceSP(int bandIndex1, int bandIndex2) {

		double[][] probDistBand1_SP = this.probabilityDistributionsSP[bandIndex1];
		double[][] probDistBand2_SP = this.probabilityDistributionsSP[bandIndex2];

		int NUM_BINS = 256;
		AtomicReference<Double> totalDistance = new AtomicReference<>(0.0);

		IntStream.range(0, this.getNumSuperpixels()).parallel().forEach(superpixelIndex -> {

//		for (int superpixelIndex = 0; superpixelIndex < this.getNumSuperpixels(); superpixelIndex++) {
			double[] probDistBand1_P = probDistBand1_SP[superpixelIndex];
			double[] probDistBand2_P = probDistBand2_SP[superpixelIndex];

			double kl = IntStream.range(0, NUM_BINS).mapToDouble(i -> {


				if (probDistBand2_P[i] == 0.0) {
					probDistBand2_P[i] = 0.00000001;
				}

				return probDistBand1_P[i] * DoubleMath.log2(probDistBand1_P[i] / probDistBand2_P[i]);
			}).sum();
			totalDistance.set(totalDistance.get() + Math.abs(kl));
//		}
		});
		return totalDistance.get();
	}

	private double calculateKlDivergenceSPmean(int bandIndex1, int bandIndex2) {
		if (this.probabilityDistributionsSPmean == null) {
			throw new IllegalStateException("ProbabilityDistributions for SP means is not calculated.");
		}
		double[] probDistBand1 = this.probabilityDistributionsSPmean[bandIndex1];
		double[] probDistBand2 = this.probabilityDistributionsSPmean[bandIndex2];

		int NUM_BINS = 256;


		double kl = IntStream.range(0, NUM_BINS).mapToDouble(i -> {
			if (probDistBand2[i] == 0.0 && probDistBand1[i] != 0.0) {
				probDistBand2[i] = 0.0000001;
			}
			if (probDistBand2[i] == 0.0 && probDistBand1[i] == 0.0) {
				return 0;
			}
			return probDistBand1[i] * DoubleMath.log2(probDistBand1[i] / probDistBand2[i]);
		}).sum();
		return kl;
	}

	/**
	 * Calculates probability distributions, using superpixel means.
	 */
	public void calculateProbabilityDistributionsSPmean() {
		log.info("Calculating probability distributions superpixelmeans for dataset {}...", this.datasetName);

		int NUM_BINS = 256;
		this.probabilityDistributionsSPmean = new double[this.numBands][NUM_BINS];

		if (this.superpixelContainer == null) {
			throw new IllegalStateException("SuperpixelContainer is not initialized.");
		}

		for (int bandIndex = 0; bandIndex < this.numBands; bandIndex++) {

			INDArray bandData = this.superpixelContainer.getSuperpixelMeans(bandIndex);

			double[] r = bandData.toDoubleVector();

			int[] histogram = new int[NUM_BINS];
			double[] normalHistogram = new double[NUM_BINS];

			double min = (double) bandData.minNumber();
			double max = (double) bandData.maxNumber();

			// Bin all superpixels to create histogram
			for (double p : r) {
				int bin = (int) Math.floor((p - min) / (max - min) * (NUM_BINS - 1));
				histogram[bin] += 1;
			}

			// Normalize histogram into probability distribution
			for (int i = 0; i < NUM_BINS; i++) {
				normalHistogram[i] = (double) histogram[i] / (double) (r.length);
			}
			this.probabilityDistributionsSPmean[bandIndex] = normalHistogram;
		}
	}

	/**
	 * Calculates joint probability distributions, using superpixel means.
	 */
	private double[][] calculateJointProbabilityDistributionSPmean(int bandIndex1, int bandIndex2) {
		INDArray bandData1 = this.superpixelContainer.getSuperpixelMeans(bandIndex1);
		INDArray bandData2 = this.superpixelContainer.getSuperpixelMeans(bandIndex2);

		double[] r1 = bandData1.toDoubleVector();
		double[] r2 = bandData2.toDoubleVector();

		int NUM_BINS = 256;

		if (this.superpixelContainer == null) {
			throw new IllegalStateException("SuperpixelContainer is not initialized.");
		}

		int[][] histogram = new int[NUM_BINS][NUM_BINS];
		double[][] normalHistogram = new double[NUM_BINS][NUM_BINS];

		double min1 = (double) bandData1.minNumber();
		double max1 = (double) bandData1.maxNumber();

		double min2 = (double) bandData2.minNumber();
		double max2 = (double) bandData2.maxNumber();

		// Bin all pixels to create histogram
		for (int i = 0; i < this.getNumSuperpixels(); i++) {
			double p1 = r1[i];
			double p2 = r2[i];
			int bin1 = (int) Math.floor((p1 - min1) / (max1 - min1) * (NUM_BINS - 1));
			int bin2 = (int) Math.floor((p2 - min2) / (max2 - min2) * (NUM_BINS - 1));
			histogram[bin1][bin2] += 1;
		}

		// Normalize histogram into probability distribution
		double sum = 0;
		for (int i = 0; i < NUM_BINS; i++) {
			for (int j = 0; j < NUM_BINS; j++) {
				if (histogram[i][j] != 0) {
					double normal = (double) histogram[i][j] / (double) (this.getNumSuperpixels());
					normalHistogram[i][j] = normal;
					sum += normal;
				}
			}
		}

		return normalHistogram;
	}

	/**
	 * Calculates probability distributions, for each superpixel.
	 */
	public void calculateProbabilityDistributionsSP() {
		log.info("Calculating probability distributions for each superpixel for dataset {}...", this.datasetName);

		int NUM_BINS = 256;
		this.probabilityDistributionsSP = new double[this.numBands][this.getNumSuperpixels()][NUM_BINS];

		INDArray SP_map = superpixelContainer.getSuperixelmap();
		if (this.superpixelContainer == null) {
			throw new IllegalStateException("SuperpixelContainer is not initialized.");
		}

//		for (int bandIndex = 0; bandIndex < this.numBands; bandIndex++) {
		IntStream.range(0, this.numBands).parallel().forEach(bandIndex -> {


			double[][] probabilityDistributionSP = new double[this.getNumSuperpixels()][NUM_BINS];
			INDArray bandData = this.data.get(NDArrayIndex.point(bandIndex), NDArrayIndex.all(), NDArrayIndex.all());
			IntStream.range(0, this.getNumSuperpixels()).forEach(superpixelIndex -> {

				INDArray result = bandData.mul(SP_map.eq(superpixelIndex));

				INDArray superpixelBandData = result.dup().reshape(numPixels);


				int[] histogram = new int[NUM_BINS];
				double[] normalHistogram = new double[NUM_BINS];

				double max = (double) superpixelBandData.maxNumber();
				BooleanIndexing.replaceWhere(superpixelBandData, 5000, Conditions.equals(0.0));
				double min = (double) superpixelBandData.minNumber();
				double[] r = superpixelBandData.toDoubleVector();

				int countp = 0;

				// Bin all superpixels to create histogram
				for (double p : r) {
					if (p != 5000) {
						int bin = (int) Math.floor((p - min) / (max - min) * (NUM_BINS - 1));
						histogram[bin] += 1;
						countp += 1;
					}
				}

				// Normalize histogram into probability distribution
				for (int i = 0; i < NUM_BINS; i++) {
					normalHistogram[i] = (double) histogram[i] / (double) (countp);
				}

				probabilityDistributionSP[superpixelIndex] = normalHistogram;
			});
			this.probabilityDistributionsSP[bandIndex] = probabilityDistributionSP;
//		}
		});
	}


	/**
	 * Calculates the Correlation Coefficients between two bands, using superpixel means.
	 *
	 * @param bandIndex1 The index of the first band.
	 * @param bandIndex2 The index of the second band.
	 * @return The Correlation Coefficient between the two bands.
	 */
	public double CorrelationCoefficientDistance_SP(int bandindex1, int bandindex2) {
		return this.correlationCoefficientsSPmean[bandindex1][bandindex2];
	}

	/**
	 * Calculates CorrelationCoefficients, using superpixel means.
	 */
	public void calculateCorrelationCoefficients_SP() {
		double[][] correlationCoefficients = new double[this.numBands][this.numBands];
		double[] bandAverage = IntStream.range(0, this.numBands).mapToDouble(b -> Arrays.stream(this.probabilityDistributionsSPmean[b]).average().getAsDouble()).toArray();

		for (int i = 0; i < this.numBands; i++) {
			for (int j = 0; j < this.numBands; j++) {
				if (i == j) {
					correlationCoefficients[i][j] = 0;
					continue;
				}

				double[] SP_MEAN_dataBandi = this.superpixelContainer.getSuperpixelMeans(i).toDoubleVector();
				double[] SP_MEAN_dataBandj = this.superpixelContainer.getSuperpixelMeans(j).toDoubleVector();

				long sumAbove = 0;
				long sumI = 0;
				long sumJ = 0;

				int numSuperPixels = this.getNumSuperpixels();
				for (int x = 0; x < numSuperPixels; x++) {
					double i_SP_MEAN = SP_MEAN_dataBandi[x] - bandAverage[i];
					double j_SP_MEAN = SP_MEAN_dataBandj[x] - bandAverage[j];
					sumAbove += i_SP_MEAN * j_SP_MEAN;
					sumI += Math.pow(i_SP_MEAN, 2);
					sumJ += Math.pow(j_SP_MEAN, 2);
				}

				correlationCoefficients[i][j] = sumAbove / (Math.sqrt(sumI) * Math.sqrt(sumJ));
			}
		}
		this.correlationCoefficientsSPmean = correlationCoefficients;
	}

	/**
	 * Calculates the Correlation Coefficients between two bands, using superpixel means.
	 *
	 * @param bandIndex1 The index of the first band.
	 * @param bandIndex2 The index of the second band.
	 * @return The Disjoint information between the two bands.
	 */
	private double disjointInfoSPmeanDistance(int bandIndex1, int bandIndex2) {
		if (this.DisjointInfosSuperpixelLevel == null) {
			throw new IllegalStateException("ProbabilityDistributions for DisjointInfoSuperpixelLevel is not calculated.");
		}
		return this.DisjointInfosSuperpixelLevel[bandIndex1][bandIndex2];
	}

	public void calculateDisjointInfoSuperpixelLevel() {
		this.DisjointInfosSuperpixelLevel = new double[this.numBands][this.numBands];
		for (int bandindex1 = 0; bandindex1 < this.numBands; bandindex1++) {
			for (int bandindex2 = 0; bandindex2 < this.numBands; bandindex2++) {
				if (this.DisjointInfosSuperpixelLevel[bandindex1][bandindex2] == 0.0) {
				}
				this.DisjointInfosSuperpixelLevel[bandindex1][bandindex2] = calculateDisjointInfoSPmean(bandindex1, bandindex2);
				this.DisjointInfosSuperpixelLevel[bandindex2][bandindex1] = this.DisjointInfosSuperpixelLevel[bandindex1][bandindex2];
			}
		}
	}


	private double calculateDisjointInfoSPmean(int bandIndex1, int bandIndex2) {

		if (this.probabilityDistributionsSPmean == null) {
			throw new IllegalStateException("ProbabilityDistributions for SPmean is not calculated.");
		}
		double[] probDistBand1 = this.probabilityDistributionsSPmean[bandIndex1];
		double[] probDistBand2 = this.probabilityDistributionsSPmean[bandIndex2];
		double[][] jointProbDistBandSPmean = this.calculateJointProbabilityDistributionSPmean(bandIndex1, bandIndex2);

		int NUM_BINS = 256;


		double di = IntStream.range(0, NUM_BINS).parallel().mapToDouble(x -> {
			return IntStream.range(0, NUM_BINS).mapToDouble(y -> {
				if (probDistBand1[x] == 0.0 && probDistBand2[y] != 0.0) {
					probDistBand1[x] = 0.0000001;
				}

				double joint = jointProbDistBandSPmean[x][y];
				return joint == 0.0 ? 1.0 : joint * DoubleMath.log2((probDistBand1[x] * probDistBand2[y]) / Math.pow(joint, 2));
			}).sum();
		}).sum();


		return di;
	}


	public double distance(DistanceMeasure distanceMeasure, int bandIndex1, int bandIndex2) {
		switch (distanceMeasure) {
			case PIXEL_EUCLIDEAN:
				return this.euclideanDistance(bandIndex1, bandIndex2);
			case SP_MEAN_EUCLIDEAN:
				return this.euclideanDistanceSP(bandIndex1, bandIndex2);
			case PIXEL_KL_DIVERGENCE:
				return this.KlDivergenceDistance(bandIndex1, bandIndex2);
			case SP_MEAN_KL_DIVERGENCE:
				return this.KlDivergenceDistanceSP(bandIndex1, bandIndex2);
			case SP_MEAN_COR_COF:
				return this.CorrelationCoefficientDistance_SP(bandIndex1, bandIndex2);
			case SP_LEVEL_KL_DIVERGENCE_L1NORM:
				return this.KlDivergencesSuperpixelLevelL1normDistance(bandIndex1, bandIndex2);
			case SP_MEAN_DISJOINT:
				return this.disjointInfoSPmeanDistance(bandIndex1, bandIndex2);
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
