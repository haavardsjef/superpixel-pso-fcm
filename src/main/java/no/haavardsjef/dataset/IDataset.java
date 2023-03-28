package no.haavardsjef.dataset;

import no.haavardsjef.utility.Bounds;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

public interface IDataset {


	double euclideanDistance(int index1, int index2);

	public INDArray getData();

	public Bounds getBounds();
}
