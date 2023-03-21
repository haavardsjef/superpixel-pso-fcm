package no.haavardsjef.utility;

import lombok.extern.log4j.Log4j2;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import us.hebi.matlab.mat.format.Mat5;
import us.hebi.matlab.mat.format.Mat5File;
import us.hebi.matlab.mat.types.Matrix;
import us.hebi.matlab.mat.types.Source;
import us.hebi.matlab.mat.types.Sources;

import java.io.IOException;

@Log4j2
public class HyperspectralDataLoader {

	public static INDArray loadData(String correctedDataPath) throws IOException {
		Source source = Sources.openFile(correctedDataPath);
		Mat5File file = Mat5.newReader(source).readMat();
		Matrix matrix = file.getArray(0);

		double[][][] dataCube = new double[matrix.getDimensions()[2]][matrix.getDimensions()[0]][matrix
				.getDimensions()[1]];

		for (int b = 0; b < matrix.getDimensions()[2]; b++) {
			for (int x = 0; x < matrix.getDimensions()[1]; x++) {
				for (int y = 0; y < matrix.getDimensions()[0]; y++) {
					dataCube[b][y][x] = matrix.getDouble(new int[]{y, x, b});
				}
			}
		}
		log.info("Successfully loaded HSI data from " + correctedDataPath);
		return Nd4j.create(dataCube);
	}

	public static INDArray loadGroundTruth(String groundTruthPath) throws IOException {
		Source source = Sources.openFile(groundTruthPath);
		Mat5File file = Mat5.newReader(source).readMat();
		Matrix matrix = file.getArray(0);
		
		int[][] dataCube = new int[matrix.getDimensions()[0]][matrix.getDimensions()[1]];
		for (int i = 0; i < matrix.getDimensions()[0]; i++) {
			for (int j = 0; j < matrix.getDimensions()[1]; j++) {
				int classLabel = (int) matrix.getDouble(i, j);
				dataCube[i][j] = classLabel;
			}
		}
		log.info("Successfully loaded ground truth data from " + groundTruthPath);
		return Nd4j.create(dataCube);
	}
}
