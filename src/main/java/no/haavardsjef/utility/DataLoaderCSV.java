package no.haavardsjef.utility;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class DataLoaderCSV implements IDataLoader {

	double[][] data;

	public DataLoaderCSV() {
		this.data = new double[1000][2];
	}

	public void loadData() {
		String path = "data/data2.csv";
		try {
			Scanner sc = new Scanner(new File(path));

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
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		System.out.println("Successfully loaded data from path: " + path);
	}

	public double[][] getData() {
		return data;
	}

	public double[] getDataPoint(int index) {
		return data[index];
	}

	public int getNumberOfDataPoints() {
		return data.length;
	}


}
