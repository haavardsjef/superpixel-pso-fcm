package no.haavardsjef.dataset;

import no.haavardsjef.utility.BenchmarkDataLoader;
import no.haavardsjef.utility.Bounds;
import no.haavardsjef.utility.DistanceMeasure;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;

public class BenchmarkDataset implements IDataset {

	private INDArray data; // Rank: 2
	private int numDataPoints;
	private String datasetPath;
	private BenchmarkDatasetName datasetName;
	private Bounds bounds;

	public BenchmarkDataset(BenchmarkDatasetName datasetName) throws IOException {
		this.datasetPath = "data/benchmarks/" + datasetName;
		this.datasetName = datasetName;
		this.load();
	}


	/**
	 * Loads the dataset from the given path.
	 *
	 * @throws IOException if the dataset cannot be loaded
	 */
	private void load() throws IOException {
		String dataset = this.datasetPath + "/data.csv";
		this.data = BenchmarkDataLoader.loadData(dataset);
		this.bounds = new Bounds(0, (int) this.data.shape()[0] - 1);
	}


	public double euclideanDistance(int index1, int index2) {
		INDArray data1 = this.data.get(NDArrayIndex.point(index1), NDArrayIndex.all());
		INDArray data2 = this.data.get(NDArrayIndex.point(index2), NDArrayIndex.all());
		return data1.distance2(data2);
	}

	@Override
	public INDArray getData() {
		return this.data;
	}

	@Override
	public Bounds getBounds() {
		return this.bounds;
	}

	@Override
	public double distance(DistanceMeasure distanceMeasure, int index1, int index2) {
		switch (distanceMeasure) {
			case PIXEL_EUCLIDEAN:
				return this.euclideanDistance(index1, index2);
			default:
				throw new IllegalArgumentException("Unknown distance measure: " + distanceMeasure);
		}
	}

	public double[][] getDataAsArray() {
		return this.data.toDoubleMatrix();
	}

}
