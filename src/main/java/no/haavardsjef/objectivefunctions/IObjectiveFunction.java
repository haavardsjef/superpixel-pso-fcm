package no.haavardsjef.objectivefunctions;

import java.util.List;

public interface IObjectiveFunction {

    public float evaluate(List<Integer> candidateSolution);
}
