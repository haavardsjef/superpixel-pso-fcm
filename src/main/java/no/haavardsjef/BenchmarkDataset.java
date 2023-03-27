package no.haavardsjef;

import no.haavardsjef.superpixelsegmentation.SuperpixelContainer;
import no.haavardsjef.utility.BenchmarkDataLoader;
import no.haavardsjef.utility.HyperspectralDataLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.util.List;

public class BenchmarkDataset implements IDataset {

	private INDArray data; // Rank: 2
	private int numDataPoints;
	private String datasetPath;
	private DatasetName datasetName;

	public BenchmarkDataset(DatasetName datasetName) throws IOException {
		this.datasetPath = "data/" + datasetName;
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
	}


	@Override
	public double euclideanDistance(int index1, int index2) {
		INDArray data1 = this.data.get(NDArrayIndex.point(index1), NDArrayIndex.all());
		INDArray data2 = this.data.get(NDArrayIndex.point(index2), NDArrayIndex.all());
		return data1.distance2(data2);
	}

	@Override
	public INDArray getData() {
		return this.data;
	}

	public static void main(String[] args) {
		try {
			BenchmarkDataset benchmarkDataset = new BenchmarkDataset(DatasetName.benchmark_problem_clustering);
			double distance = benchmarkDataset.euclideanDistance(0, 1);
			System.out.println(distance);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
