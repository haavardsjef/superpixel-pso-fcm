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
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

@Log4j2
public class BenchmarkDataLoader {

	public static INDArray loadData(String dataPath) throws IOException {

//		double[][] data = new double[1000][2];
		List<List<Double>> data = new ArrayList<>();

		Scanner sc = new Scanner(new File(dataPath));

		// Skip header line
		sc.nextLine();

		// Read data to array
		int i = 0;
		while (sc.hasNextLine()) {
			List<Double> dataPoint = new ArrayList<>();

			String[] line = sc.nextLine().split(",");
			for (int j = 0; j < line.length; j++) {
				dataPoint.add(Double.parseDouble(line[j]));
			}
			data.add(dataPoint);
			i++;
		}
		sc.close();


		log.info("Successfully loaded benchmark data from " + dataPath);
		double[][] dataAsArray = data.stream().map(u -> u.stream().mapToDouble(d -> d).toArray()).toArray(double[][]::new);
		return Nd4j.create(dataAsArray);
	}
}
