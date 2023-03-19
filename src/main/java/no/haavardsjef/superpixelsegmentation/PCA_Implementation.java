package no.haavardsjef.superpixelsegmentation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nd4j.linalg.factory.Nd4j;

public abstract class PCA_Implementation {
	// https://javadoc.io/doc/org.nd4j/nd4j-api/latest/org/nd4j/linalg/dimensionalityreduction/PCA.html
	// https://deeplearning4j.konduit.ai/nd4j/tutorials/quickstart

	/**
	 * Performs PCA on the data, reducing the dimensionality to 3.
	 *
	 * @param hsiDataFlattened - The flattened band data, the first index is the band, the second index is the flattened pixel index.
	 * @return INDArray with the principle components.
	 */
	public static INDArray performPCA(double[][] hsiDataFlattened, boolean normalize) {
		INDArray dataset = Nd4j.create(hsiDataFlattened);
		dataset = dataset.transpose();


		INDArray principleComponents = PCA.pca(dataset, 3, normalize);

		return principleComponents;

	}

}
