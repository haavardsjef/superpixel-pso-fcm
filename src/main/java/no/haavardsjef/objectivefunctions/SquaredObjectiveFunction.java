package no.haavardsjef.objectivefunctions;

import no.haavardsjef.objectivefunctions.IObjectiveFunction;

import java.util.List;

public class SquaredObjectiveFunction implements IObjectiveFunction {
    @Override
    public float evaluate(List<Integer> candidateSolution) {
        return candidateSolution.stream().map(cs -> cs * cs).reduce(0, (a, b) -> a + b);
    }
}
