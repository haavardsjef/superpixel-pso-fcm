package no.haavardsjef.vizualisation;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import no.haavardsjef.pso.Particle;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class PlotLine {


	public void plot(double[] data) {
		// Create x and y lists
		List<Double> x = new ArrayList<>();
		List<Double> y = new ArrayList<>();

		for (int i = 0; i < data.length; i++) {
			x.add((double) i);
			y.add(data[i]);
		}


		// Plot swarm using matplotlib4j
		Plot plt = Plot.create();
		plt.plot().add(x, y);

		try {
			plt.show();
		} catch (IOException | PythonExecutionException e) {
			e.printStackTrace();
		}
	}


}
