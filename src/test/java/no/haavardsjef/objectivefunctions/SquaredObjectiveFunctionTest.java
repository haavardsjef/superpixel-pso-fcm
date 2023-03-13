package no.haavardsjef.objectivefunctions;

import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.List;

public class SquaredObjectiveFunctionTest extends TestCase {

    public void testEvaluate() {
        List<Integer> candidateSolution = new ArrayList<Integer>();
        candidateSolution.add(1);
        candidateSolution.add(2);
        candidateSolution.add(3);
        candidateSolution.add(4);
        candidateSolution.add(5);

        SquaredObjectiveFunction squaredFitnessFunction = new SquaredObjectiveFunction();
        assertEquals(55.0f, squaredFitnessFunction.evaluate(candidateSolution));

    }
}