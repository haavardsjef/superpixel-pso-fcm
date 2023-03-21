package no.haavardsjef;

import lombok.extern.log4j.Log4j2;
import no.haavardsjef.superpixelsegmentation.SuperpixelContainer;
import no.haavardsjef.utility.HyperspectralDataLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;

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

	public void setupSuperpixelContainer() {
		this.superpixelContainer = new SuperpixelContainer(this.data);
		this.superpixelContainer.generateSuperpixelMap();
	}


	public static void main(String[] args) throws IOException {
		Dataset ds = new Dataset("data/indian_pines", DatasetName.indian_pines);
		ds.setupSuperpixelContainer();
		ds.superpixelContainer.calculateSuperpixelMeans();
	}
}
