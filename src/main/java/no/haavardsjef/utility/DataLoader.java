package no.haavardsjef.utility;

import us.hebi.matlab.mat.format.Mat5;
import us.hebi.matlab.mat.format.Mat5File;
import us.hebi.matlab.mat.types.Matrix;
import us.hebi.matlab.mat.types.Source;
import us.hebi.matlab.mat.types.Sources;

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
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


	/**
	 * @return The flattened band data, the first index is the band, the second index is the flattened pixel index.
	 */
	public double[][] getDataFlatted() {
		return dataFlatted;
	}

	/**
	 * @param index - the index of the datapoint
	 * @return The pixel values for a single band
	 */
	public double[] getDataPoint(int index) {
		return dataFlatted[index];
	}

	/**
	 * @return The number of datapoints, in this case the number of bands available in the HSI.
	 */
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

	/**
	 *  Returns an array of arrays that contains the data values for selected bands, for all pixels.
	 *  The index in the outer array corresponds to which pixel in the flattened image.
	 *  The entries in the inner array contain to the data value of the corresponding band from the bands input parameter.
	 *  Example: getPixelValuesForBands(Arrays.asList([10, 20]) - returns an array with as many elements as there are pixels,
	 *  each element is an array [x1, x2] where x1 is the value of band 10 in that pixel, and x2 is the value of band 20 in that pixel.
	 * @param bands - The selected bands that we should use
	 * @return - Array containing an array with the band values of the selected bands for the pixel of ith in
	 */
	public double[][] getPixelValuesForBands(List<Integer> bands){
		double[][] pixelValuesForBands = new double[dataFlatted[0].length][bands.size()];
		for (int i = 0; i < dataFlatted[0].length; i++) {
			for (int j = 0; j < bands.size(); j++) {
				pixelValuesForBands[i][j] = dataFlatted[bands.get(j)][i];
			}
		}
		return pixelValuesForBands;
	}

}
