package no.haavardsjef.vizualisation;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import no.haavardsjef.pso.Particle;
import no.haavardsjef.utility.Bounds;
import no.haavardsjef.App;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Visualizations {

    public static void plotSwarm(List<Particle> particles, int iteration, Bounds bounds) {
        List<Double> x = new ArrayList<>();
        List<Double> y = new ArrayList<>();

        for (Particle particle : particles) {
            x.add((double) particle.getPosition()[0]);
            y.add((double) particle.getPosition()[1]);
        }

        // Plot swarm using matplotlib4j
        Plot plt = Plot.create();
        plt.plot().add(x, y, "o");
        plt.xlim(bounds.lower(), bounds.upper());
        plt.ylim(bounds.lower(), bounds.upper());
        plt.title("Iteration " + iteration);
//            plt.show();
        plt.savefig(App.DIR + "/viz/" + iteration + ".png");
        try {
            plt.executeSilently();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (PythonExecutionException e) {
            throw new RuntimeException(e);
        }
        System.out.println("Saved plot " + iteration);

    }
}
