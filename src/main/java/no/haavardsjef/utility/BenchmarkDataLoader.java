package no.haavardsjef.utility;

import lombok.extern.log4j.Log4j2;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import us.hebi.matlab.mat.format.Mat5;
import us.hebi.matlab.mat.format.Mat5File;
import us.hebi.matlab.mat.types.Matrix;
import us.hebi.matlab.mat.types.Source;
import us.hebi.matlab.mat.types.Sources;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;

@Log4j2
public class BenchmarkDataLoader {

	public static INDArray loadData(String dataPath) throws IOException {

		double[][] data = new double[1000][2];

		Scanner sc = new Scanner(new File(dataPath));

		// Skip header line
		sc.nextLine();

		// Read data to array
		int i = 0;
		while (sc.hasNextLine()) {
			String[] line = sc.nextLine().split(",");
			for (int j = 0; j < line.length; j++) {
				data[i][j] = Double.parseDouble(line[j]);
			}
			i++;
		}
		sc.close();


		log.info("Successfully loaded benchmark data from " + dataPath);
		return Nd4j.create(data);
	}
}
