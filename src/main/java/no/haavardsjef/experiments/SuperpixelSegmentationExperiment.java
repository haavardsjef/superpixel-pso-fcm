package no.haavardsjef.experiments;

import no.haavardsjef.superpixelsegmentation.PCA_Implementation;
import no.haavardsjef.utility.DataLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

public class SuperpixelSegmentationExperiment implements IExperiment {
    @Override
    public void runExperiment() {

        DataLoader dl = new DataLoader();
        dl.loadData();
        double[][] hsiDataFlattened = dl.getDataFlatted();

        PCA_Implementation pca = new PCA_Implementation();
        INDArray principleComponents = pca.performPCA(hsiDataFlattened);
        System.out.println(principleComponents);

    }

    public static void main(String[] args) {
        SuperpixelSegmentationExperiment superpixelSegmentationExperiment = new SuperpixelSegmentationExperiment();
        superpixelSegmentationExperiment.runExperiment();
    }
}
