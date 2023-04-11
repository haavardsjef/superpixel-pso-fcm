package no.haavardsjef.dataset;

import no.haavardsjef.utility.Bounds;
import no.haavardsjef.utility.DistanceMeasure;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

public interface IDataset {


	public INDArray getData();

	public Bounds getBounds();

	double distance(DistanceMeasure distanceMeasure, int index1, int index2);
}
