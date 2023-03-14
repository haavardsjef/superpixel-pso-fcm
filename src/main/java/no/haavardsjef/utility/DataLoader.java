package no.haavardsjef.utility;

import us.hebi.matlab.mat.format.Mat5;
import us.hebi.matlab.mat.format.Mat5File;
import us.hebi.matlab.mat.types.Matrix;
import us.hebi.matlab.mat.types.Source;
import us.hebi.matlab.mat.types.Sources;

import java.io.IOException;
import java.util.HashSet;
import java.util.Optional;
import java.util.Set;

public class DataLoader implements IDataLoader {

	private double[][][] data;
	private double[][] dataFlatted; // Flattened pixel data

	private int[] groundTruthFlattened;

	public void loadData() {
		String path = "data/indian_pines_corrected.mat";
		try {
			Source source = Sources.openFile(path);
			Mat5File file = Mat5.newReader(source).readMat();
			Matrix matrix = file.getArray(0);

			double[][][] dataCube = new double[matrix.getDimensions()[2]][matrix.getDimensions()[0]][matrix
					.getDimensions()[1]];
			this.dataFlatted = new double[matrix.getDimensions()[2]][matrix.getDimensions()[0] * matrix.getDimensions()[1]];

			for (int b = 0; b < matrix.getDimensions()[2]; b++) {
				for (int x = 0; x < matrix.getDimensions()[1]; x++) {
					for (int y = 0; y < matrix.getDimensions()[0]; y++) {
						dataCube[b][y][x] = matrix.getDouble(new int[]{y, x, b});
						dataFlatted[b][y * matrix.getDimensions()[1] + x] = matrix.getDouble(new int[]{y, x, b});
					}
				}
			}

			this.data = dataCube;
			System.out.println("Successfully loaded data from path: " + path);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}


	public double[] getDataPoint(int index) {
		return dataFlatted[index];
	}

	public int getNumberOfDataPoints() {
		return dataFlatted.length;
	}

	// Load ground truth
	public void loadGroundTruth() {
		String path = "data/indian_pines_gt.mat";
		try (Source source = Sources.openFile(path)) {
			Mat5File file = Mat5.newReader(source).readMat();

			Matrix matrix = file.getArray(0);
			Set<Integer> classes = new HashSet<>();
			this.groundTruthFlattened = new int[matrix.getDimensions()[0] * matrix.getDimensions()[1]];
			for (int i = 0; i < matrix.getDimensions()[0]; i++) {
				for (int j = 0; j < matrix.getDimensions()[1]; j++) {
					int classLabel = (int) matrix.getDouble(i, j);
					this.groundTruthFlattened[i * matrix.getDimensions()[1] + j] = classLabel;
					classes.add(classLabel);
				}
			}
			System.out.println("Successfully loaded ground truth from path: " + path);

		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public int[] getGroundTruthFlattened() {
		return groundTruthFlattened;
	}

}
