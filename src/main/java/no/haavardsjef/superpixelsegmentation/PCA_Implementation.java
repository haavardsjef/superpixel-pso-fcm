package no.haavardsjef.superpixelsegmentation;

import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class PCA_Implementation
{
    // https://javadoc.io/doc/org.nd4j/nd4j-api/latest/org/nd4j/linalg/dimensionalityreduction/PCA.html
    // https://deeplearning4j.konduit.ai/nd4j/tutorials/quickstart

    public INDArray performPCA(double[][] hsiDataFlattened) {
        INDArray dataset = Nd4j.create(hsiDataFlattened);
        dataset = dataset.transpose();

        INDArray principleComponents = PCA.pca(dataset, 3, false);

        return principleComponents;

    }

}
