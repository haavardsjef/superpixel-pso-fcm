package no.haavardsjef.vizualisation;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import com.github.sh0nk.matplotlib4j.builder.HistBuilder;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public abstract class PlotLine {


	public static void plot(List<Double> data) {
		// Create x and y lists
		List<Double> x = new ArrayList<>();

		for (int i = 0; i < data.size(); i++) {
			x.add((double) i);
		}


		// Plot swarm using matplotlib4j
		Plot plt = Plot.create();
		plt.plot().add(x, data);

		try {
			plt.show();
		} catch (IOException | PythonExecutionException e) {
			e.printStackTrace();
		}
	}

	public static void plotMultiple(List<Double>... data) {
		// Create x and y lists

		Plot plt = Plot.create();
		for (List<Double> list : data) {
			List<Double> x = new ArrayList<>();

			for (int i = 0; i < list.size(); i++) {
				x.add((double) i);
			}
			plt.plot().add(x, list);
		}


		// Plot swarm using matplotlib4j

		try {
			plt.show();
		} catch (IOException | PythonExecutionException e) {
			e.printStackTrace();
		}
	}

	public static void barChart(List<Double> data) {
		// Create x and y lists
		List<Double> x = new ArrayList<>();


		List<String> xLabels = IntStream.range(1, data.size() + 1)
				.mapToObj(Integer::toString)
				.collect(Collectors.toList());

		// Plot swarm using matplotlib4j
		Plot plt = Plot.create();
		HistBuilder histBuilder = plt.hist();
		histBuilder.bins(data.size());
		histBuilder.add(data);

		try {
			plt.show();
		} catch (IOException | PythonExecutionException e) {
			e.printStackTrace();
		}
	}


	public static void saveToCsv(List<Double>... data) throws IOException {
		FileWriter csvWriter = new FileWriter("plot.csv");

		// Write header row
		csvWriter.append("List Index");
		for (int i = 0; i < data.length; i++) {
			csvWriter.append(",List " + i);
		}
		csvWriter.append("\n");

		// Write data rows
		int numLists = data.length;
		int maxLength = 0;
		for (int i = 0; i < numLists; i++) {
			if (data[i].size() > maxLength) {
				maxLength = data[i].size();
			}
		}
		for (int row = 0; row < maxLength; row++) {
			csvWriter.append(Integer.toString(row));
			for (int col = 0; col < numLists; col++) {
				if (row < data[col].size()) {
					csvWriter.append("," + data[col].get(row).toString());
				} else {
					csvWriter.append(",");
				}
			}
			csvWriter.append("\n");
		}

		csvWriter.flush();
		csvWriter.close();
	}


}
